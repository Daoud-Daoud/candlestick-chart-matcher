import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
from multiprocessing import Pool, cpu_count
from functools import partial
import numba
import time

# --- Configuration ---
NUM_CORES = max(1, cpu_count() - 1)  # Leave one core free

@numba.jit(nopython=True)
def fast_dtw(seq1, seq2):
    """
    Optimized DTW implementation using Numba
    """
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return 0
    
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq1[i-1] - seq2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )
    
    max_possible_distance = max(np.ptp(seq1), np.ptp(seq2)) * max(n, m)
    if max_possible_distance == 0:
        return 100
    
    similarity = 100 * (1 - dtw_matrix[n, m] / max_possible_distance)
    return max(0, min(100, similarity))

def calculate_enhanced_similarity(user_line, candidate_line, weights=None):
    """
    Calculate multiple similarity metrics and combine them
    """
    if weights is None:
        weights = {
            'cosine': 0.3,
            'pearson': 0.2,
            'amplitude': 0.25,
            'dtw': 0.25
        }
    
    user_flat = user_line.flatten()
    candidate_flat = candidate_line.flatten()
    
    # Vectorized operations
    cosine_sim = cosine_similarity(user_flat.reshape(1, -1), 
                                 candidate_flat.reshape(1, -1))[0][0]
    cosine_score = ((cosine_sim + 1) / 2) * 100
    
    try:
        pearson_corr, _ = pearsonr(user_flat, candidate_flat)
        pearson_score = ((pearson_corr + 1) / 2) * 100 if not np.isnan(pearson_corr) else 0
    except:
        pearson_score = 0
    
    user_range = np.ptp(user_flat)
    candidate_range = np.ptp(candidate_flat)
    
    user_std = np.std(user_flat)
    candidate_std = np.std(candidate_flat)
    
    range_similarity = 100 * (1 - abs(user_range - candidate_range) / max(user_range, candidate_range)) if max(user_range, candidate_range) > 1e-6 else 100
    std_similarity = 100 * (1 - abs(user_std - candidate_std) / max(user_std, candidate_std)) if max(user_std, candidate_std) > 1e-6 else 100
        
    amplitude_score = (range_similarity + std_similarity) / 2
    amplitude_score = max(0, min(100, amplitude_score))
    
    # Use optimized DTW
    dtw_score = fast_dtw(user_flat, candidate_flat)
    
    final_score = (
        weights['cosine'] * cosine_score +
        weights['pearson'] * pearson_score +
        weights['dtw'] * dtw_score +
        weights['amplitude'] * amplitude_score
    )
    
    return {
        'final_score': final_score,
        'cosine': cosine_score,
        'pearson': pearson_score,
        'amplitude': amplitude_score,
        'dtw': dtw_score,
        'user_range': user_range,
        'candidate_range': candidate_range
    }

def extract_candle_centers_from_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}. Check file path and integrity.")
        return None

    # Convert to HSV for color detection
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define color ranges for red and green candles
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    # Create masks for red and green candles
    mask_red = cv2.add(cv2.inRange(hsv_img, lower_red1, upper_red1),
                       cv2.inRange(hsv_img, lower_red2, upper_red2))
    mask_green = cv2.inRange(hsv_img, lower_green, upper_green)

    # Morphological operations to clean up the masks
    kernel = np.ones((3, 3), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

    # Find contours in the masks
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract candle centers
    candles = []
    for cnt in contours_red + contours_green:
        if cv2.contourArea(cnt) > 5:  # Filter small noise
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 0 and 0.1 < h/w < 10:  # Filter non-candle shapes
                center_x = x + w / 2
                center_y = y + h / 2
                candles.append((center_x, center_y))

    # Sort candles by x-coordinate
    candles = sorted(candles, key=lambda c: c[0])

    if len(candles) < 2:
        print("Not enough candles detected to form a pattern.")
        return None

    # Normalize y-coordinates
    centers_y = np.array([c[1] for c in candles])
    y_min = centers_y.min()
    y_max = centers_y.max()
    y_range = y_max - y_min if y_max != y_min else 1e-6

    normalized_centers = [(c[0], (y_max - c[1]) / y_range) for c in candles]

    return np.array([c[1] for c in normalized_centers])

def pattern_vector_from_centers(centers):
    return np.array(centers).reshape(1, -1)

def _match_worker(args, user_line, center_arr, pattern_len, candles_after, min_match_percent):
    """
    Optimized worker function for pattern matching
    """
    i = args  # Simplify argument passing
    
    centers = center_arr[i:i+pattern_len]
    
    # Skip if we don't have enough data points
    if len(centers) < pattern_len:
        return None
    
    # Vectorized normalization
    min_y = centers.min()
    max_y = centers.max()
    y_range = max_y - min_y if max_y != min_y else 1e-6
    norm_centers = (centers - min_y) / y_range

    similarity_metrics = calculate_enhanced_similarity(user_line, norm_centers)

    if similarity_metrics['final_score'] >= min_match_percent:
        return {
            'index': i,
            'match_percent': similarity_metrics['final_score'],
            'similarity_breakdown': similarity_metrics,
            'norm_centers': norm_centers,
            'raw_centers': centers
        }
    return None

def find_matching_shape_lines_enhanced(df, user_line, pattern_len, candles_after, min_match_percent):
    """
    Optimized pattern matching with better multiprocessing
    """
    total_len = len(df)
    print(f"Searching through {total_len - pattern_len - candles_after} potential patterns...")

    if 'Center' not in df.columns:
        raise ValueError("DataFrame must have a 'Center' column. Call process_historical_data first.")
    
    center_arr = df['Center'].values

    # Create partial function with fixed arguments
    worker = partial(_match_worker, 
                    user_line=user_line,
                    center_arr=center_arr,
                    pattern_len=pattern_len,
                    candles_after=candles_after,
                    min_match_percent=min_match_percent)

    # Split work into chunks for better parallel processing
    chunk_size = max(1000, (total_len - pattern_len - candles_after) // (NUM_CORES * 10))
    indices = range(total_len - pattern_len - candles_after)
    
    with Pool(NUM_CORES) as pool:
        # Process in chunks to avoid memory issues
        raw_results = []
        for chunk_start in range(0, len(indices), chunk_size):
            chunk = indices[chunk_start:chunk_start + chunk_size]
            chunk_results = [res for res in pool.map(worker, chunk) if res is not None]
            raw_results.extend(chunk_results)

    # Process matches
    matches_with_data = []
    for res in raw_results:
        i = res['index']
        segment = df.iloc[i:i+pattern_len]
        subsequent = df.iloc[i+pattern_len:i+pattern_len+candles_after]
        
        res.update({
            'pattern_data': segment,
            'subsequent_data': subsequent,
        })
        matches_with_data.append(res)
    
    # Sort by match percentage
    matches_with_data.sort(key=lambda m: m['match_percent'], reverse=True)

    # Filter overlapping patterns
    final_matches = []
    seen_indices = set()
    for match in matches_with_data:
        start_idx = match['index']
        end_idx = match['index'] + len(match['pattern_data']) + len(match['subsequent_data']) - 1

        # Check for overlap
        if not any(i in seen_indices for i in range(start_idx, end_idx + 1)):
            final_matches.append(match)
            seen_indices.update(range(start_idx, end_idx + 1))

    print(f"Found {len(final_matches)} unique matches above {min_match_percent}% threshold")
    return final_matches

def process_historical_data(df):
    """
    Optimized data processing
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Vectorized center calculation
    df['Center'] = (df['Open'] + df['Close']) / 2
    
    # Ensure we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")
    
    return df

def plot_user_line_overlay(ax, pattern_data, norm_centers, user_centers):
    """
    Optimized plotting function
    """
    x_vals = np.arange(len(pattern_data))
    price_range = pattern_data['High'].max() - pattern_data['Low'].min()
    y_vals = norm_centers * price_range + pattern_data['Low'].min()
    user_y_vals = user_centers * price_range + pattern_data['Low'].min()
    
    ax.plot(x_vals, user_y_vals, color='orange', linestyle='--', linewidth=2, label='User Pattern Line')
    ax.legend()

def print_detailed_match_info(match, match_num):
    """
    Optimized info printing
    """
    metrics = match['similarity_breakdown']
    print(f"\n--- Match {match_num} Details ---")
    print(f"Date: {match['pattern_data'].index[0]}")
    print(f"Overall Score:      {metrics['final_score']:.2f}%")
    print(f"Shape (Cosine):     {metrics['cosine']:.1f}%")
    print(f"Correlation:        {metrics['pearson']:.1f}%")
    print(f"Amplitude Match:    {metrics['amplitude']:.1f}%")
    print(f"Pattern Alignment:  {metrics['dtw']:.1f}%")
    
    if len(match['subsequent_data']) > 0:
        first_close = match['pattern_data']['Close'].iloc[-1]
        final_close = match['subsequent_data']['Close'].iloc[-1]
        change_pct = ((final_close - first_close) / first_close) * 100
        print(f"Subsequent Move:    {change_pct:+.2f}%")

def summarize_outcomes(matches, html_format=False):
    """
    Calculate statistics for ALL matches found above threshold
    """
    ups = downs = flats = total_change = 0
    for match in matches:
        if len(match['subsequent_data']) < 1:
            continue
        first_close = match['pattern_data']['Close'].iloc[-1]
        final_close = match['subsequent_data']['Close'].iloc[-1]
        change_pct = ((final_close - first_close) / first_close) * 100
        total_change += change_pct

        if change_pct > 0.1:
            ups += 1
        elif change_pct < -0.1:
            downs += 1
        else:
            flats += 1

    total = len(matches)
    analyzed = ups + downs + flats
    
    if analyzed == 0:
        return "<p style='color:red;'>No valid matches found for outcome analysis.</p>" if html_format else "No valid matches found for outcome analysis."

    bullish_pct = 100 * ups / analyzed if analyzed > 0 else 0
    bearish_pct = 100 * downs / analyzed if analyzed > 0 else 0
    flat_pct = 100 * flats / analyzed if analyzed > 0 else 0
    avg_move = total_change / analyzed if analyzed > 0 else 0

    if html_format:
        return f"""
        <h2 style='color:#fbb034;'>📊 Pattern Outcome Summary</h2>
        <table style='margin:auto; color:#eee; background:#2b2b2b; border-collapse:collapse; font-size:16px;'>
            <tr><th style='padding:8px 16px; border:1px solid #555;'>Total Matches Found</th><td style='padding:8px 16px; border:1px solid #555;'>{total}</td></tr>
            <tr><th style='padding:8px 16px; border:1px solid #555;'>📈 Bullish</th><td style='padding:8px 16px; border:1px solid #555;'>{ups} ({bullish_pct:.1f}%)</td></tr>
            <tr><th style='padding:8px 16px; border:1px solid #555;'>📉 Bearish</th><td style='padding:8px 16px; border:1px solid #555;'>{downs} ({bearish_pct:.1f}%)</td></tr>
            <tr><th style='padding:8px 16px; border:1px solid #555;'>⚖️ Neutral</th><td style='padding:8px 16px; border:1px solid #555;'>{flats} ({flat_pct:.1f}%)</td></tr>
            <tr><th style='padding:8px 16px; border:1px solid #555;'>📊 Avg Move</th><td style='padding:8px 16px; border:1px solid #555;'>{avg_move:+.3f}%</td></tr>
        </table>
        """
    else:
        return f"""
        ==========================================
        PATTERN OUTCOME SUMMARY
        ==========================================
        Total Matches Found: {total}
        Bullish Outcomes: {ups} ({bullish_pct:.1f}%)
        Bearish Outcomes: {downs} ({bearish_pct:.1f}%)
        Neutral Outcomes: {flats} ({flat_pct:.1f}%)
        Average Subsequent Move: {avg_move:+.3f}%
        """
