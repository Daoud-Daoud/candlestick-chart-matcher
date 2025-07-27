from flask import Flask, request, render_template
import os
from collections import defaultdict
from utils.run_match import run_pattern_match
import time

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
DATA_FOLDER = 'data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cache for assets and timeframes
_assets_timeframes_cache = None
_last_cache_update = 0
CACHE_TIMEOUT = 3600  # 1 hour cache

def get_available_assets_and_timeframes(data_dir=DATA_FOLDER):
    global _assets_timeframes_cache, _last_cache_update
    
    current_time = time.time()
    if _assets_timeframes_cache and (current_time - _last_cache_update) < CACHE_TIMEOUT:
        return _assets_timeframes_cache
    
    available = defaultdict(list)
    for fname in os.listdir(data_dir):
        if fname.endswith('.csv') and '_' in fname:
            asset, tf = fname.replace('.csv', '').split('_', 1)
            available[asset].append(tf)
    
    _assets_timeframes_cache = dict(available)
    _last_cache_update = current_time
    return _assets_timeframes_cache

@app.route("/", methods=["GET", "POST"])
def index():
    assets_timeframes = get_available_assets_and_timeframes()

    if request.method == "POST":
        start_time = time.time()
        
        uploaded_file = request.files.get('pattern')
        threshold = float(request.form.get('threshold', 90))
        top_n = int(request.form.get('top_n', 5))
        asset = request.form.get('asset')
        timeframe = request.form.get('timeframe')

        if not asset or not timeframe:
            return render_template("index.html", matches=None, assets_timeframes=assets_timeframes)

        if uploaded_file:
            # Save uploaded file
            save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(save_path)

            data_path = f"{DATA_FOLDER}/{asset}_{timeframe}.csv"

            if not os.path.exists(data_path):
                return render_template("index.html", matches=None, 
                                     summary="❌ Data file not found.", 
                                     assets_timeframes=assets_timeframes)

            matches, summary = run_pattern_match(save_path, data_path, threshold, top_n)
            
            # Add processing time to summary
            processing_time = time.time() - start_time
            summary += f"\n\n⏱ Processing time: {processing_time:.2f} seconds"

            #add Data available range (drom date x to date y) according to searched file
            with open(data_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    date_range = f"Data available from {lines[1].split(',')[0]} to {lines[-1].split(',')[0]}"
                    summary += f"\n\n📅 {date_range}"
                    
            
            relative_paths = ["/" + path.replace("\\", "/") for path in matches]
            return render_template("index.html", 
                                matches=relative_paths, 
                                summary=summary, 
                                assets_timeframes=assets_timeframes)

    return render_template("index.html", matches=None, assets_timeframes=assets_timeframes)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
