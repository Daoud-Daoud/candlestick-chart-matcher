# Candlestick Chart Matcher

This repository contains code designed to identify candlestick chart patterns within historical financial data. The core idea is to allow users to upload a screenshot of a candlestick chart for a specific asset, and then the application will find similar patterns in historical data.

## How it Works

The logic analyzes the uploaded chart by spotting the center of each candlestick. It then creates a line connecting these centers, taking their relative Y-positions from the image and normalizing them by dividing by the maximum Y-value. This normalized sequence is stored as an array.

Recognized candlestick patterns and their normalized arrays are then matched with historical data. The application loops through the historical data, candle by candle, using the maximum value for normalization. Arrays from the historical data are compared with the image array, and a similarity percentage is calculated.

A key feature is the ability to retrieve historical chart data that most closely matches your uploaded chart. This allows you to observe what happened historically after a similar pattern occurred, providing insights based on the assumption that history may repeat itself.

## How to Use the App

### Installation

1.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download or Clone the Repository:**
    ```bash
    git clone [https://github.com/Daoud-Daoud/candlestick-chart-matcher.git](https://github.com/Daoud-Daoud/candlestick-chart-matcher.git)
    ```

### Optional: Add More Data

You can extend the historical data by adding CSV files to the `data` folder. Ensure that the CSV headers follow this specific order and labels: `Date | Open | High | Low | Close | Volume`.
*Note: The `Volume` and other labels are currently not used in the matching process.*

### Run the App Locally

1.  Navigate into the repository folder.
2.  Open your command prompt or terminal.
3.  Run the application:
    ```bash
    python app.py
    ```
4.  Ctrl+Click (or Cmd+Click) on the local address displayed in the terminal, usually `127.0.0.1:5000`, to open the app in your web browser.

### Fill the Form

Once the application is running in your browser, you will see a form with the following fields:

* **Choose File:** Upload your candlestick chart screenshot.
    * *Best Practice:* Include only red and green candlesticks, without price lines or other chart overlays for optimal results.
* **Matching Threshold (%):** Set the minimum percentage similarity required for a match to be considered.
* **Number of Top Matches to Show:** Specify how many of the most similar historical matches you want to see. These will be drawn as output, ordered by their matching percentage.
* **Select Asset:** Choose from the available assets to search through.
* **Select Timeframe:** Select the timeframe (e.g., daily, hourly).
    * *Note:* A lower timeframe will generally result in longer processing times due to more data points.

### Hit "Find Matches"

After filling out the form and clicking "Find Matches," the application will process your request. Once the search is finalized, the matching historical charts will be displayed, along with a small table of statistics.
 
  




https://github.com/user-attachments/assets/8cb22f6f-c214-446e-b6ed-40084c3a6aae


### Connect with Me

<a href="https://www.linkedin.com/in/daoud-daoud/" target="_blank">LinkedIn</a> |
<a href="https://github.com/Daoud-Daoud" target="_blank">GitHub</a> |
<a href="mailto:daoud.tradting7@gmail.com">Email</a>

