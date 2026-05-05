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
from dataclasses import dataclass, field
from typing import Any
import cv2
import numpy as np

log = logging.getLogger(__name__)

FAR = "far"
NEAR = "near"

SETS_TO_WIN = 2
GAMES_TO_WIN = 6
TIEBREAK_PTS = 7

# Si on fait pour les doubles, il faudra rajouter des paramètres pour les supertiebreaks


@dataclass
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
            self.point_won(self.other(self.server_side), frame, double_fault=True)
            return True

    def point_won(self, winner: str, frame: int, double_fault: bool = False):
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

    def run(self, frames):
        for frame in frames:
            if self.score.match_over():
                break
            self.process_frame(frame)

        if self.rally_ongoing and frames:
            last_frame = frames[-1]["frame_idx"]
            self.finish_rally(last_frame)

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
        # Close the previous rally before starting a new point
        if self.rally_ongoing and not self.is_serving:
            self.finish_rally(frame_index)

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
        if not self.rally_ongoing:
            return

        ball = frame.get("ball")
        court_kps = frame.get("court_keypoints")

        if not ball or ball.get("conf", 0) < 0.2:
            log.warning(f"Frame {frame_index}: Action-spotting detected bounce, but tracking missed the ball.")
            
        if self.is_serving and bounce_side == self.serve_side:
            is_double_fault = self.score.serve_fault(frame_index)
            if is_double_fault:
                self.reset_rally_state()
            else:
                # First fault: go inert until second serve
                self.rally_ongoing = False
                self.is_serving = False
                self.last_swing_side = None
            return

        is_out = False

        if ball and court_kps and len(court_kps) == 14:
            ball_pt = (ball["x"], ball["y"])
            center_x = (court_kps[12][0] + court_kps[13][0]) / 2.0

            if self.is_serving:
                # 1. SERVE VALIDATION
                # The net is not explicitly in the 14 keypoints, but it lies exactly halfway
                # between the top baselines (4, 6) and bottom baselines (5, 7).
                net_left = ((court_kps[4][0] + court_kps[5][0]) / 2.0, (court_kps[4][1] + court_kps[5][1]) / 2.0)
                net_right = ((court_kps[6][0] + court_kps[7][0]) / 2.0, (court_kps[6][1] + court_kps[7][1]) / 2.0)

                if bounce_side == "far":
                    # FAR service boxes: bounded by top service line (8, 9) and the net
                    service_poly = np.array([court_kps[8], court_kps[9], net_right, net_left], dtype=np.float32)
                else:
                    # NEAR service boxes: bounded by bottom service line (10, 11) and the net
                    service_poly = np.array([court_kps[10], court_kps[11], net_right, net_left], dtype=np.float32)
                
                hull = cv2.convexHull(service_poly)
                # measureDist=True allows us to see exactly how far out the ball is
                dist = cv2.pointPolygonTest(hull, ball_pt, measureDist=True)

                bounced_right = ball["x"] > center_x
                expected_right = (self.expected_serve_target_x == "right")
                
                # Give a 10-pixel margin of error for tracking jitter
                if dist < -10 or (bounced_right != expected_right):
                    is_out = True

            else:
                # 2. RALLY VALIDATION (Singles Court)
                singles_indices = [4, 5, 6, 7]
                singles_corners = [court_kps[i] for i in singles_indices]
                pts = np.array(singles_corners, dtype=np.float32)
                hull = cv2.convexHull(pts)
                dist = cv2.pointPolygonTest(hull, ball_pt, measureDist=True)

                if dist < -10:
                    is_out = True

        if is_out:
            if self.is_serving:
                is_double_fault = self.score.serve_fault(frame_index)
                if is_double_fault:
                    self.reset_rally_state()
                else:
                    self.rally_ongoing = False
                    self.is_serving = False
                    self.last_swing_side = None
                return
            else:
                if self.last_swing_side is not None:
                    # Safely determine the winner without risking an AttributeError
                    winner = "far" if self.last_swing_side == "near" else "near"
                    self.finish_rally(frame_index, winner_side=winner)
                    return

        self.is_serving = False

        if self.last_swing_side is not None and bounce_side == self.last_swing_side:
            winner = "far" if self.last_swing_side == "near" else "near"
            self.finish_rally(frame_index, winner_side=winner)
            return

    def swing(self, swing_side: str, frame_index: int, frame: dict):
        if not frame.get("ball") or frame["ball"].get("conf", 0) < 0.2:
            log.warning(f"Frame {frame_index}: Action-spotting detected swing, but tracking missed the ball.")
            
        if self.rally_ongoing:
            # SAFETY CATCH: If the receiver swings, the serve landed safely and the rally is on.
            # This fixes the state if action-spotting completely missed the serve bounce!
            if self.is_serving and swing_side != self.serve_side:
                self.is_serving = False
                
            self.last_swing_side = swing_side

    def finish_rally(self, frame_index: int,
                     winner_side: str|None = None):
        if winner_side is None:
            winner_side = (self.other(self.last_swing_side)
                           if self.last_swing_side else FAR)
            
        self.score.point_won(winner_side, frame_index)
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
        }