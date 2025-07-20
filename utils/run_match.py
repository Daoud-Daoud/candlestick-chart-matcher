import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from .matcher import (
    extract_candle_centers_from_image,
    pattern_vector_from_centers,
    process_historical_data,
    find_matching_shape_lines_enhanced,
    plot_user_line_overlay,
    summarize_outcomes,
    print_detailed_match_info
)

# Constants
NUM_MATCHES_TO_SHOW = 5
CANDLES_AFTER_PATTERN = 15
UPLOAD_DIR = 'static/uploads'

os.makedirs(UPLOAD_DIR, exist_ok=True)

def run_pattern_match(user_image_path, data_path='data/XAU.csv', min_match_percent=90, top_n=5):
    image_paths = []
    summary_output = []

    # Step 1: Extract pattern
    user_centers = extract_candle_centers_from_image(user_image_path)
    if user_centers is None:
        return [], "Could not extract pattern from the uploaded image."

    pattern_len = len(user_centers)
    user_vector = pattern_vector_from_centers(user_centers)

    # Step 2: Load historical data
    if not os.path.exists(data_path):
        return [], "Historical data file not found."

    df = pd.read_csv(data_path, index_col=0)
    df.columns = [c.capitalize() for c in df.columns]
    df = process_historical_data(df)

    # Step 3: Find matches
    matches = find_matching_shape_lines_enhanced(
        df, user_vector, pattern_len, CANDLES_AFTER_PATTERN, min_match_percent
    )

    if not matches:
        return [], "No matches found above threshold. Try adjusting settings."

    # Step 4: Generate match plots
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    style = mpf.make_mpf_style(marketcolors=mc)

    for i, match in enumerate(matches[:top_n]):
        combined = pd.concat([match['pattern_data'], match['subsequent_data']])
        pattern_len_plot = len(match['pattern_data'])

        fig, ax = mpf.plot(
            combined,
            type='candle',
            style=style,
            ylabel='Price',
            returnfig=True,
            figsize=(14, 6),
            title=f"Match {i+1} | Score: {match['match_percent']:.2f}%"
        )

        ax[0].axvline(x=pattern_len_plot - 0.5, color='blue', linestyle='--', linewidth=2)
        ax[0].text(pattern_len_plot - 0.5, combined['High'].max(), ' Pattern End',
                  color='blue', verticalalignment='top', fontweight='bold')
        plot_user_line_overlay(ax[0], match['pattern_data'], match['norm_centers'], user_centers)

        output_path = os.path.join(UPLOAD_DIR, f'match_{i+1}.png')
        fig.savefig(output_path)
        plt.close(fig)
        image_paths.append(output_path)

        # Capture brief info for each match
        metrics = match['similarity_breakdown']
        change_pct = 0
        if len(match['subsequent_data']) > 0:
            first_close = match['pattern_data']['Close'].iloc[-1]
            final_close = match['subsequent_data']['Close'].iloc[-1]
            change_pct = ((final_close - first_close) / first_close) * 100

        summary_output.append(
            f"Match {i+1}: {match['pattern_data'].index[0].date()}\n"
            f"Score: {match['match_percent']:.2f}% | Move: {change_pct:+.2f}%\n"
        )

    # Step 5: Add overall summary
    outcome_html = summarize_outcomes(matches, html_format=True)
    summary_output.append(outcome_html)



    return image_paths, "\n".join(summary_output)