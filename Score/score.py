"""
Score computation for tennis matches.

Architecture
------------
The module has two layers:

  TennisScore   — pure tennis scoring rules (no video awareness).
                  Knows about points, games, sets, deuce, advantage,
                  tiebreaks, and change-of-ends.  Fully unit-testable
                  without any video data.

  ScoreComputer — video-aware wrapper.  Consumes the merged per-frame
                  list produced by io_utils.merge_frames, translates
                  action-spotting events into TennisScore calls, and
                  assembles the final result dict.

Event -> scoring logic
-----------------------
The six action-spotting classes and what they signal:

    far_court_serve   / near_court_serve
        -> start of a new point; server = the player on that court side

    far_court_bounce  / near_court_bounce
        -> ball landed on that side of the court.
           A bounce on the SERVER's own side right after a serve = fault.
           A bounce on the HITTER's own side during a rally = ball went
           into the net or out -> that player loses the point.

    far_court_swing   / near_court_swing
        -> a player struck the ball; used to track who hit last.

Point-ending detection
-----------------------
The model does not label point endings directly.  A point ends when:

  1. A new serve is detected -> previous rally is over.
     Winner = player who did NOT swing last (i.e. the last hitter lost).
  2. A bounce lands on the hitter's own side during a rally
     -> ball went out or into the net.
  3. End of the clip -> last rally closed with the same heuristic as (1).
  4. Double fault -> server loses the point.
"""

from __future__ import annotations

import logging
from typing import Any
import cv2
import numpy as np

log = logging.getLogger(__name__)

FAR = "far"
NEAR = "near"

SETS_TO_WIN = 2
GAMES_TO_WIN = 6
TIEBREAK_PTS = 7

# To account for duplicate events, if an event happens within this number of frames after the pending, it is ignored
PENDING_END_GRACE_FRAMES = 8
PENDING_SERVE_FAULT_GRACE_FRAMES = 8


def _kps_have(kps, *needed):
    return (kps is not None
            and len(kps) == 14
            and all(kps[i] is not None for i in needed))



class TennisScore:
    """
    Tennis score tracker.
    """
    def __init__(self, server_side=FAR, initial_sets=None, initial_games=None, initial_points=None):
        self.server_side = server_side

        # Different possibilities for the points in a game
        self.points = initial_points or {FAR: 0, NEAR: 0}
        self.deuce = (self.points[FAR] >= 3 and self.points[NEAR] >= 3) # Deuce is true for 40-40 and advantages
        self.advantage_to = None

        # Different possibilities for games and sets
        self.games = initial_games or {FAR: 0, NEAR: 0}
        self.sets = initial_sets or {FAR: 0, NEAR: 0}
        self.in_tiebreak = (self.games[FAR] == GAMES_TO_WIN and self.games[NEAR] == GAMES_TO_WIN)
        
        # To know if first or second serve
        self.serve_number = 1

        # time of the start of a point
        self.point_start_frame = 0
        self.point_log = []
        self.game_log = []
        self.set_log = []

    def other(self, curr_side: str) -> str:
        """
        Returns the other side.
        FAR bacomes NEAR 
        NEAR becomes FAR
        """
        if curr_side == NEAR:
            return FAR
        elif curr_side == FAR:
            return NEAR
        else:
            print("Side provided in function other isn't NEAR or FAR")
            return

    def new_point(self, server_side: str, frame: int):
        """
        Change the variables of the class when a new point is starting
        """
        self.server_side = server_side
        self.point_start_frame = frame
        # first serve is reset in point_won

    def serve_fault(self, frame: int) -> bool:
        """
        Use when a serve fault appears.
        Returns True and updates the score if it was a double fault 
        Returns False and register to second serve if it was the first serve.
        """
        if self.serve_number == 1:
            self.serve_number = 2
            return False
        else:
            self.point_won(self.other(self.server_side), frame, double_fault=True, reason="double_fault")
            return True

    def point_won(self, winner: str, frame: int, double_fault: bool = False, reason: str = "unknown"):
        """
        Register the player that won and update the score.
        """
        # Need to reset the serve if there were two serves this point
        serve_number_this_point = self.serve_number
        self.serve_number = 1

        if self.in_tiebreak:
            self.tiebreak_point(winner)
        else:
            self.regular_point(winner)

        self.point_log.append({
            'start_frame': self.point_start_frame,
            'end_frame': frame,
            'winner_side': winner,
            'server_side': self.server_side,
            'serve_number': serve_number_this_point,
            'double_fault': double_fault,
            'reason': reason,
            'score_after': {'points': dict(self.points), 'games': dict(self.games), 'sets': dict(self.sets)},
        })

    def regular_point(self, winner: str) -> None:
        """
        Updates the score if we are in a regular setting (not a tiebreak)
        """
        loser = self.other(winner)
        if self.deuce:
            # If winner had the advantage, wins the game
            if self.advantage_to == winner:
                self.advantage_to = None
                self.deuce = False
                self.game_won(winner)
            # If loser had the advantage, back to 40-40
            elif self.advantage_to == loser:
                self.advantage_to = None
            # If 40-40, advantage to winner
            else:
                self.advantage_to = winner
        else:
            self.points[winner] += 1
            winner_points, loser_points = self.points[winner], self.points[loser]

            # Check if someone won the game
            if winner_points >= 3 and (winner_points - loser_points) >= 2:
                self.game_won(winner)
            # Check if becomes deuce (40-40)
            elif winner_points == 3 and loser_points == 3:
                self.deuce = True

    def tiebreak_point(self, winner: str):
        """
        Updates the score for tiebreak points (first to 7)
        """
        self.points[winner] += 1
        winner_points, loser_points = self.points[winner], self.points[self.other(winner)]
        
        # Rotate server every 2 points (after point 1, then every 2)
        total = winner_points + loser_points
        if total % 2 == 1:  # odd total = time to switch
            self.server_side = self.other(self.server_side)
        
        if winner_points >= TIEBREAK_PTS and (winner_points - loser_points) >= 2:
            self.game_won(winner)

    def game_won(self, winner: str):
        """
        Updates the score when someone wins a game
        """
        self.points = {FAR: 0, NEAR: 0}
        self.deuce = False
        self.advantage_to = None
        self.games[winner] += 1
        loser = self.other(winner)
        winner_games, loser_games = self.games[winner], self.games[loser]

        self.game_log.append({
            'winner_side': winner,
            'games_after': dict(self.games),
            'sets_after': dict(self.sets),
        })

        if self.in_tiebreak and winner_games > loser_games:
            self.in_tiebreak = False
            self.set_won(winner)
        elif winner_games == GAMES_TO_WIN and loser_games == GAMES_TO_WIN:
            self.in_tiebreak = True
        elif winner_games >= GAMES_TO_WIN and (winner_games - loser_games) >= 2:
            self.set_won(winner)

    def set_won(self, winner: str):
        games_before_reset = dict(self.games)
        self.sets[winner] += 1
        self.games = {FAR: 0, NEAR: 0}

        self.set_log.append({
            'winner_side': winner,
            'games': {'far': games_before_reset[FAR], 'near': games_before_reset[NEAR]},
        })


    def match_over(self) -> bool:
        return max(self.sets.values()) >= SETS_TO_WIN

    def match_winner(self) -> str | None:
        if self.sets[FAR] >= SETS_TO_WIN: return FAR
        if self.sets[NEAR] >= SETS_TO_WIN: return NEAR
        return None

    def score_string(self) -> str:
        parts = [f"{s['games'][FAR]}-{s['games'][NEAR]}" for s in self.set_log]
        if self.games[FAR] or self.games[NEAR]:
            parts.append(f"{self.games[FAR]}-{self.games[NEAR]}")
        return ", ".join(parts) if parts else "0-0"



class ScoreComputer:
    """
    Computes tennis scores from a video.
    """

    def __init__(self, tracking_meta: dict[str, Any], initial_sets=None, initial_games=None, initial_points=None):
        self.meta = tracking_meta
        self.fps = tracking_meta.get("fps", 30.0)
        # Faudra mettre tout les arguments du init les même que tennisScore mais j'ai la flemme
        self.score = TennisScore(
            initial_sets=initial_sets,
            initial_games=initial_games,
            initial_points=initial_points,
        )

        # rally state
        self.rally_ongoing = False
        self.last_swing_side = None
        self.serve_side = None

        self.is_serving = False   # True during the time between the serve and the first bounce, false otherwise, it is to account for the fact that there are two serves
        self.expected_serve_target_x = None
        self.frame_states = [] 
        self.pending_point_end = None
        self.pending_serve_fault = None
        self.observed_server_x_side = None
        self.bounce_log = []

    @property
    def in_point(self) -> bool:
        """True whenever a point has started (1st or 2nd serve) but not yet been decided."""
        return self.rally_ongoing or self.is_serving or self.score.serve_number > 1

    def has_pending_point_end(self) -> bool:
        return self.pending_point_end is not None

    def _observe_server_x_side(self, frame: dict, server_side: str) -> str | None:
        """
        Return 'left' or 'right' (camera-x) for where the player on `server_side`
        is standing this frame, or None if court keypoints/players unavailable.
        """
        court_kps = frame.get("court_keypoints")
        if not _kps_have(court_kps, 12, 13):
            return None
        center_x = (court_kps[12][0] + court_kps[13][0]) / 2.0

        server_label = "top" if server_side == FAR else "bottom"
        server = next((p for p in frame.get("players", [])
                    if p.get("side") == server_label), None)
        if server is None:
            return None
        x1, _, x2, _ = server["bbox"]
        return "left" if (x1 + x2) / 2.0 < center_x else "right"

    def propose_point_end(
        self,
        frame_index: int,
        winner_side: str | None = None,
        reason: str = "unknown",
        evidence: dict | None = None,
    ) -> None:
        """
        Register a tentative point end.

        Important:
        This does NOT update TennisScore yet.

        The point will be committed only if:
          - a new serve is detected, or
          - the clip ends before more rally actions appear.

        If more rally actions appear before the next serve, this pending end is cancelled.
        """
        if winner_side is None:
            winner_side = (
                self.score.other(self.last_swing_side)
                if self.last_swing_side
                else FAR
            )

        # If we already have a pending point end, keep the first one.
        # The earliest plausible ending is usually the best candidate.
        if self.pending_point_end is not None:
            return

        self.pending_point_end = {
            "frame_idx": frame_index,
            "winner_side": winner_side,
            "reason": reason,
            "evidence": evidence or {},
        }

        log.info(
            "Frame %s: proposed pending point end: winner=%s reason=%s evidence=%s",
            frame_index,
            winner_side,
            reason,
            evidence,
        )

    def commit_pending_point_end(self, commit_frame: int | None = None) -> bool:
        """
        Commit the pending point end to TennisScore.

        Returns True if something was committed.
        """
        if self.pending_point_end is None:
            return False

        pending = self.pending_point_end
        end_frame = pending["frame_idx"]

        self.score.point_won(
            pending["winner_side"],
            end_frame,
            reason=pending["reason"],
        )

        log.info(
            "Frame %s: committed pending point end from frame %s: winner=%s reason=%s",
            commit_frame,
            pending["frame_idx"],
            pending["winner_side"],
            pending["reason"],
        )

        self.pending_point_end = None
        self.reset_rally_state()
        return True

    def cancel_pending_point_end(self, frame_index: int, reason: str) -> bool:
        """
        Cancel a tentative point end because the rally appears to continue.

        Returns True if something was cancelled.
        """
        if self.pending_point_end is None:
            return False

        log.info(
            "Frame %s: cancelled pending point end from frame %s because %s",
            frame_index,
            self.pending_point_end["frame_idx"],
            reason,
        )

        self.pending_point_end = None

        # The rally is still alive.
        self.rally_ongoing = True
        self.is_serving = False
        return True

    def maybe_cancel_pending_point_end(
        self,
        frame_index: int,
        event_kind: str,
        event_side: str,
    ) -> None:
        """
        If a non-serve rally action appears after a tentative point end,
        assume the point did not actually end and cancel the pending end.

        We ignore events very close to the pending end because those are often
        duplicate detections of the same bounce/swing.
        """
        if self.pending_point_end is None:
            return

        pending_frame = self.pending_point_end["frame_idx"]
        dt = frame_index - pending_frame

        if dt <= PENDING_END_GRACE_FRAMES:
            return

        if event_kind in {"swing", "bounce"}:
            self.cancel_pending_point_end(
                frame_index,
                reason=f"rally_continued_with_{event_side}_{event_kind}",
            )


    def propose_serve_fault(self, frame_index: int, evidence: dict | None = None) -> None:
        """
        Tentatively register a serve fault. Does NOT bump serve_number yet.
        Committed only when:
        - a same-server, same-x-side serve appears (real 2nd serve), or
        - the clip ends.
        Cancelled when:
        - a receiver-side swing/bounce appears after the grace window (rally continued), or
        - a serve appears from a different x-side or different server (we missed a point ending).
        """
        if self.pending_serve_fault is not None:
            return  # keep first

        self.pending_serve_fault = {
            "frame_idx":     frame_index,
            "server_side":   self.serve_side,
            "server_x_side": self.observed_server_x_side,
            "evidence":      evidence or {},
        }
        log.info("Frame %s: proposed pending serve fault — server=%s x_side=%s",
                frame_index, self.serve_side, self.observed_server_x_side)


    def commit_pending_serve_fault(self, commit_frame: int) -> bool:
        """Commit to TennisScore. Returns True iff this commit was a double fault."""
        if self.pending_serve_fault is None:
            return False
        p = self.pending_serve_fault
        self.pending_serve_fault = None
        is_double_fault = self.score.serve_fault(p["frame_idx"])
        log.info("Frame %s: committed pending serve fault from frame %s (double=%s)",
                commit_frame, p["frame_idx"], is_double_fault)
        if is_double_fault:
            self.reset_rally_state()
        return is_double_fault


    def cancel_pending_serve_fault(self, frame_index: int, reason: str) -> bool:
        if self.pending_serve_fault is None:
            return False
        log.info("Frame %s: cancelled pending serve fault from frame %s — %s",
                frame_index, self.pending_serve_fault["frame_idx"], reason)
        self.pending_serve_fault = None
        # The 1st serve was actually OK; we're effectively still in the rally.
        self.rally_ongoing = True
        self.is_serving = False
        return True


    def maybe_cancel_pending_serve_fault(
        self, frame_index: int, event_kind: str, event_side: str
    ) -> None:
        """Receiver-side swing/bounce after grace → rally continued, cancel."""
        if self.pending_serve_fault is None:
            return
        if frame_index - self.pending_serve_fault["frame_idx"] <= PENDING_SERVE_FAULT_GRACE_FRAMES:
            return
        receiver_side = self.score.other(self.pending_serve_fault["server_side"])
        if event_kind in {"swing", "bounce"} and event_side == receiver_side:
            self.cancel_pending_serve_fault(
                frame_index,
                reason=f"rally_continued_with_{event_kind}_on_receiver_side",
            )

    def run(self, frames):
        for frame in frames:
            if self.score.match_over():
                break
            self.process_frame(frame)

            self.frame_states.append({
                "frame_idx":    frame["frame_idx"],
                "in_point":     self.in_point,
                "serve_number": self.score.serve_number,
                "server_side":  self.score.server_side,
                "pending_point_end": self.pending_point_end is not None,
                "pending_point_end_frame": (
                    self.pending_point_end["frame_idx"]
                    if self.pending_point_end else None
                ),
                "pending_point_end_reason": (
                    self.pending_point_end["reason"]
                    if self.pending_point_end else None
                ),
                "pending_serve_fault":         self.pending_serve_fault is not None,
                "pending_serve_fault_frame":   (self.pending_serve_fault["frame_idx"]
                                                if self.pending_serve_fault else None),
                "pending_serve_fault_x_side":  (self.pending_serve_fault["server_x_side"]
                                                if self.pending_serve_fault else None),
            })

        if frames:
            last_frame = frames[-1]["frame_idx"]

            if self.pending_serve_fault is not None:
                self.commit_pending_serve_fault(commit_frame=last_frame)
            elif self.pending_point_end is not None:
                self.commit_pending_point_end(commit_frame=last_frame)

            elif self.rally_ongoing:
                self.finish_rally(last_frame, reason="end_of_clip")

        return self.build_result()


    def process_frame(self, frame) -> None:
        for event in frame["events"]:
            frame_index = frame["frame_idx"]
            label = event["label"]
            side = label.split("_court_")[0]
            kind = label.split("_court_")[1]

            if kind == "serve":
                self.serve(side, frame_index, frame)
            elif kind == "bounce":
                self.bounce(side, frame_index, frame)
            elif kind == "swing":
                self.swing(side, frame_index, frame)

    def serve(self, serve_side: str, frame_index: int, frame: dict):

        observed_x_side = self._observe_server_x_side(frame, serve_side)

        if self.pending_serve_fault is not None:
            prev = self.pending_serve_fault
            same_server = (serve_side == prev["server_side"])
            x_known = (observed_x_side is not None and prev["server_x_side"] is not None)
            same_x = x_known and (observed_x_side == prev["server_x_side"])

            if same_server and (same_x or not x_known):
                # Real second serve (or we can't tell — defer to the safer commit)
                self.commit_pending_serve_fault(frame_index)
            else:
                self.cancel_pending_serve_fault(
                    frame_index,
                    reason=("server_changed" if not same_server else "x_side_flipped"),
                )

        if self.pending_point_end is not None:
            self.commit_pending_point_end(commit_frame=frame_index)

        # Close the previous rally before starting a new point
        elif self.rally_ongoing and not self.is_serving:  
            self.finish_rally(frame_index, reason="new_serve_starting")

        self.observed_server_x_side = observed_x_side

        total_points = self.score.points["far"] + self.score.points["near"]
        is_even_score = (total_points % 2 == 0)
        if serve_side == "far":
            self.expected_serve_target_x = "right" if is_even_score else "left"
        else:
            self.expected_serve_target_x = "left" if is_even_score else "right"

        if not frame.get("ball") or frame["ball"].get("conf", 0) < 0.2:
            log.warning(f"Frame {frame_index}: serve detected but ball not tracked.")

        if self.score.serve_number == 1:
            self.score.new_point(serve_side, frame_index)

        self.is_serving = True
        self.serve_side = serve_side
        self.last_swing_side = serve_side
        self.rally_ongoing = True

    def bounce(self, bounce_side: str, frame_index: int, frame: dict):
        
        self.maybe_cancel_pending_point_end(
            frame_index,
            event_kind="bounce",
            event_side=bounce_side,
        )

        if not self.rally_ongoing:
            return

        ball = frame.get("ball")
        court_kps = frame.get("court_keypoints")

        if not ball or ball.get("conf", 0) < 0.2:
            log.warning(f"Frame {frame_index}: Action-spotting detected bounce, but tracking missed the ball.")
            
        ball_pt = (ball["x"], ball["y"]) if ball else None
        if self.is_serving and bounce_side == self.serve_side:
            self.propose_serve_fault(frame_index, evidence={
                "event": "bounce_on_server_side",
                "bounce_side": bounce_side,
                "ball": ball_pt,
            })
            self.rally_ongoing = False
            self.is_serving = False
            self.last_swing_side = None
            return

        is_out = False
        if ball_pt and self.is_serving:
            # 1. SERVE VALIDATION
            # The net is not explicitly in the 14 keypoints, but it lies exactly halfway
            # between the top baselines (4, 6) and bottom baselines (5, 7).
            serve_line = [8, 9] if bounce_side == "far" else [10, 11]
            if _kps_have(court_kps, 4, 5, 6, 7, 12, 13, *serve_line):
                net_center = ((court_kps[12][0] + court_kps[13][0]) / 2.0, (court_kps[12][1] + court_kps[13][1]) / 2.0)
                net_left = ((court_kps[4][0] + court_kps[5][0]) / 2.0, (court_kps[4][1] + court_kps[5][1]) / 2.0)
                net_right = ((court_kps[6][0] + court_kps[7][0]) / 2.0, (court_kps[6][1] + court_kps[7][1]) / 2.0)

                if bounce_side == "far": # Serve came from near
                    p_left, p_center, p_right = court_kps[8], court_kps[12], court_kps[9]
                else: # Serve came from far
                    p_left, p_center, p_right = court_kps[10], court_kps[13], court_kps[11]

                if self.expected_serve_target_x == "right":
                    # Box bounded by center, right, net_right, net_center
                    service_poly = np.array([p_center, p_right, net_right, net_center], dtype=np.float32)
                else:
                    # Box bounded by left, center, net_center, net_left
                    service_poly = np.array([p_left, p_center, net_center, net_left], dtype=np.float32)
                
                hull = cv2.convexHull(service_poly)
                # measureDist=True allows us to see exactly how far out the ball is
                dist = cv2.pointPolygonTest(hull, ball_pt, measureDist=True)

                # bounced_right = ball["x"] > center_x
                # expected_right = (self.expected_serve_target_x == "right")
                
                # Give a 10-pixel margin of error for tracking jitter
                if dist < -3: # or (bounced_right != expected_right):
                    is_out = True

        elif ball_pt:
            if _kps_have(court_kps, 4, 5, 6, 7):
                # 2. RALLY VALIDATION (Singles Court)
                singles_indices = [4, 5, 6, 7]
                singles_corners = [court_kps[i] for i in singles_indices]
                pts = np.array(singles_corners, dtype=np.float32)
                hull = cv2.convexHull(pts)
                dist = cv2.pointPolygonTest(hull, ball_pt, measureDist=True)

                if dist < -10:
                    is_out = True

        self.bounce_log.append({
            "frame_idx": frame_index,
            "side": bounce_side,           # "far" or "near"
            "is_out": bool(is_out),
            "dist": float(dist) if 'dist' in locals() else None,
            "context": "serve" if self.is_serving else ("rally" if self.rally_ongoing else "idle"),
            "ball_xy": [ball["x"], ball["y"]] if ball_pt else None,
        })

        if is_out:
            if self.is_serving:
                # is_double_fault = self.score.serve_fault(frame_index)
                # if is_double_fault:
                #     self.reset_rally_state()
                # else:
                self.propose_serve_fault(frame_index, evidence={
                    "event": "ball_out_on_serve",
                    "bounce_side": bounce_side,
                    "ball": ball_pt,
                })
                self.rally_ongoing = False
                self.is_serving = False
                self.last_swing_side = None
                return
            else:
                if self.last_swing_side is not None:
                    # Safely determine the winner without risking an AttributeError
                    winner = "far" if self.last_swing_side == "near" else "near"
                    self.propose_point_end(
                        frame_index, 
                        winner_side=winner, 
                        reason="ball_out", 
                        evidence={
                            "event": "bounce",
                            "bounce_side": bounce_side,
                            "last_swing_side": self.last_swing_side,
                            "ball": ball_pt,
                        },
                    )
                    return

        self.is_serving = False

        # 
        if self.last_swing_side is not None and bounce_side == self.last_swing_side:
            winner = "far" if self.last_swing_side == "near" else "near"
            self.propose_point_end(
                frame_index,
                winner_side=winner,
                reason="bounce_on_hitter_side",
                evidence={
                    "event": "bounce",
                    "bounce_side": bounce_side,
                    "last_swing_side": self.last_swing_side,
                    "ball": ball_pt,
                },
            )
            return

        # NEW: bounce in court on the side opposite the last hitter.
        # In a normal rally this is followed by a receiver swing, which
        # will cancel this proposal via maybe_cancel_pending_point_end.
        # If no follow-up event arrives before the next serve, the
        # proposal commits — that's exactly a winner or an ace.
        if self.last_swing_side is not None and bounce_side != self.last_swing_side:
            self.propose_point_end(
                frame_index,
                winner_side=self.last_swing_side,
                reason="winner_unreturned",
                evidence={
                    "event": "bounce",
                    "bounce_side": bounce_side,
                    "last_swing_side": self.last_swing_side,
                    "ball": ball_pt,
                },
            )

    def swing(self, swing_side: str, frame_index: int, frame: dict):
        if not frame.get("ball") or frame["ball"].get("conf", 0) < 0.2:
            log.warning(f"Frame {frame_index}: Action-spotting detected swing, but tracking missed the ball.")
            
        self.maybe_cancel_pending_point_end(
            frame_index,
            event_kind="swing",
            event_side=swing_side,
        )

        self.maybe_cancel_pending_serve_fault(frame_index, "swing", swing_side)

        if self.rally_ongoing:
            # SAFETY CATCH: If the receiver swings, the serve landed safely and the rally is on.
            # This fixes the state if action-spotting completely missed the serve bounce!
            if self.is_serving and swing_side != self.serve_side:
                self.is_serving = False
                
            self.last_swing_side = swing_side

    def finish_rally(self, frame_index: int,
                     winner_side: str|None = None, reason: str = "unknown"):
        if winner_side is None:
            winner_side = (self.other(self.last_swing_side)
                           if self.last_swing_side else FAR)
            
        self.score.point_won(winner_side, frame_index, reason=reason)
        self.reset_rally_state()

    def reset_rally_state(self):
        self.rally_ongoing = False
        self.last_swing_side = None
        self.is_serving = False

    def other(self, curr_side: str):
        """
        Returns the other side.
        FAR bacomes NEAR 
        NEAR becomes FAR
        """
        if curr_side == NEAR:
            return FAR
        elif curr_side == FAR:
            return NEAR
        else:
            print("Side provided in function other isn't NEAR or FAR")
            return


    def build_result(self) -> dict[str, Any]:
        score = self.score
        ids = self.meta.get("main_player_ids", [None, None])
        side_to_id = {
            NEAR: ids[0] if len(ids) > 0 else None,
            FAR:  ids[1] if len(ids) > 1 else None,
        }

        points = [
            {
                "point_idx":    i,
                "start_frame":  p["start_frame"],
                "end_frame":    p["end_frame"],
                "winner_side":  p["winner_side"],
                "winner_id":    side_to_id.get(p["winner_side"]),
                "server_side":  p["server_side"],
                "server_id":    side_to_id.get(p["server_side"]),
                "serve_number": p["serve_number"],
                "double_fault": p["double_fault"],
                "reason":       p.get("reason", "unknown"),
                "score_after":  p["score_after"],
            }
            for i, p in enumerate(score.point_log)
        ]

        games = [
            {
                "game_idx":    i,
                "winner_side": g["winner_side"],
                "games_far":   g["games_after"][FAR],
                "games_near":  g["games_after"][NEAR],
                "sets_far":    g["sets_after"][FAR],
                "sets_near":   g["sets_after"][NEAR],
            }
            for i, g in enumerate(score.game_log)
        ]

        sets = [
            {
                "set_idx":     i,
                "winner_side": s["winner_side"],
                "score_far":   s["games"][FAR],
                "score_near":  s["games"][NEAR],
            }
            for i, s in enumerate(score.set_log)
        ]

        return {
            "video":        self.meta.get("video", ""),
            "fps":          self.fps,
            "match_winner": score.match_winner(),
            "final_score":  score.score_string(),
            "sets_far":     score.sets[FAR],
            "sets_near":    score.sets[NEAR],
            "points":       points,
            "games":        games,
            "sets":         sets,
            "frame_states": self.frame_states,
            "bounces":      self.bounce_log,
        }