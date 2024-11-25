from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import numpy as np
import time
import json


class GameInterface:
    def __init__(self, config_path="../config/config.json"):
        """
        Initialize the browser, handle cookies, and navigate to the 2048 game using configurations.
        """
        # Load configuration
        with open(config_path, "r") as config_file:
            config = json.load(config_file)

        self.move_delay = config.get("move_delay", 0.2)
        self.driver_path = config["webdriver_path"]
        self.game_url = config["game_url"]

        # Initialize the Edge browser driver
        service = Service(self.driver_path)
        self.driver = webdriver.Edge(service=service)

        # Set the browser window size
        self.driver.set_window_size(800, 600)

        # Open the game URL
        self.driver.get(self.game_url)
        time.sleep(2)  # Allow the page to load

        # Handle the cookies banner
        self._handle_cookies()

    def _handle_cookies(self):
        """
        Dismiss the cookie banner by sending the Esc key.
        """
        try:
            # Send the Esc key to dismiss the cookie banner
            self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
            time.sleep(1)  # Wait for the banner to disappear
            print("Cookie banner dismissed with Esc key.")
        except Exception as e:
            print("Error dismissing cookie banner:", e)

    def perform_move(self, direction):
        """
        Simulate a move in the given direction.
        Args:
            direction (str): One of "up", "down", "left", "right".
        """
        direction_map = {
            "up": Keys.ARROW_UP,
            "down": Keys.ARROW_DOWN,
            "left": Keys.ARROW_LEFT,
            "right": Keys.ARROW_RIGHT
        }
        if direction in direction_map:
            # Send the keypress directly to the webpage
            self.driver.find_element(By.TAG_NAME, "body").send_keys(direction_map[direction])
            time.sleep(self.move_delay)  # Add a delay to simulate human-like interaction
        else:
            raise ValueError("Invalid move direction. Use 'up', 'down', 'left', or 'right'.")

    def get_current_score(self):
        """
        Extract the current game score from the game interface.
        Returns:
            int: Current game score.
        """
        try:
            score_element = self.driver.find_element(By.CLASS_NAME, "score-container")
            current_score = int(score_element.text.split()[0])  # Extract the score text
            return current_score
        except Exception as e:
            print(f"Error fetching current score: {e}")
            return 0

    def check_game_over(self):
        """
        Check if the game is over by analyzing move possibilities.
        Returns:
            bool: True if no valid moves are left, False otherwise.
        """
        for move in ["up", "down", "left", "right"]:
            try:
                self.perform_move(move)
                return False  # A valid move was possible
            except ValueError:
                continue
        return True  # No moves left

    def reset(self):
        """
        Reset the game by refreshing the page and extracting the initial board.
        Returns:
            np.array: The initial board state.
        """
        self.driver.refresh()
        time.sleep(2)  # Allow the page to load
        from src.board_utils import extract_board
        return np.array(extract_board(self.driver)).flatten()

    def step(self, action):
        """
        Execute an action and return the next state, reward, and done status.
        Args:
            action (str): Action to perform ("up", "down", "left", "right").
        Returns:
            next_state (np.array): Flattened board after the move.
            reward (float): Change in score or custom reward.
            done (bool): True if the game is over.
        """
        from src.board_utils import extract_board

        self.perform_move(action)  # Perform the move
        next_state = np.array(extract_board(self.driver)).flatten()  # Get new board state
        reward = self.get_current_score()  # Get reward based on score
        done = self.check_game_over()  # Check if the game is over

        return next_state, reward, done

    def close(self):
        """
        Close the browser.
        """
        self.driver.quit()


