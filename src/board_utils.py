from selenium.webdriver.common.by import By

def extract_board(driver):
    """
    Extract the current board state from the game.
    Args:
        driver: Selenium WebDriver instance.

    Returns:
        A 2D list representing the board state.
    """
    # Initialize an empty 4x4 board

    board = [[0 for _ in range(4)] for _ in range(4)]
    tiles = driver.find_elements(By.CLASS_NAME, "tile")

    for tile in tiles:
        classes = tile.get_attribute("class").split()
        value = int([cls.split("-")[1] for cls in classes if cls.startswith("tile-")][0])

        # Extract the row and column explicitly
        position = [cls for cls in classes if cls.startswith("tile-position-")][0]
        position_parts = position.replace("tile-position-", "").split("-")  # Remove prefix and split
        row, col = int(position_parts[0]) - 1, int(position_parts[1]) - 1  # Convert to 0-index

        board[row][col] = value

    return board