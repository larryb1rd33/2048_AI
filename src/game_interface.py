from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import json
import time

class GameInterface:
    def __init__(self, config_path="../config/config.json"):
        """
        Initialize the browser, handle cookies, and navigate to the 2048 game using configurations.
        """
        # Load configuration
        with open(config_path, "r") as config_file:
            config = json.load(config_file)

        self.driver_path = config["webdriver_path"]
        self.browser = config["browser"]
        self.game_url = config["game_url"]

        # Initialize the Edge browser driver
        service = Service(self.driver_path)
        self.driver = webdriver.Edge(service=service)

        # Open the game URL
        self.driver.get(self.game_url)
        time.sleep(2)  # Allow time for the page to load

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
        else:
            raise ValueError("Invalid move direction. Use 'up', 'down', 'left', or 'right'.")

    def close(self):
        """
        Close the browser.
        """
        self.driver.quit()


# Example usage
if __name__ == "__main__":
    game = GameInterface()

    try:
        # Perform some moves
        game.perform_move("up")
        game.perform_move("right")
        game.perform_move("down")
        game.perform_move("left")
    finally:
        print("Closing the game...")
        time.sleep(2)
        game.close()