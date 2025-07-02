from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load your model (adjust path as needed)
model = pickle.load(open("mvp_model.pkl", "rb"))

# Your existing function to get predictions
def predict_mvp(year):
    data = pd.read_csv("player_data.csv")
    print("CSV columns:", data.columns.tolist())

    # Filter for the selected year
    season_data = data[data["Year"] == int(year)].copy()
    print(f"Found {len(season_data)} players for {year}")

    # Use the exact features your model was trained on
    feature_columns = [
        'Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P',
        '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB',
        'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Year',
        'W', 'L', 'W/L%', 'GB', 'PS/G',
        'PA/G', 'SRS', 'PTS_R', 'AST_R', 'STL_R', 'BLK_R', '3P_R']
    features = season_data[feature_columns]

    # Make predictions
    season_data["Prediction"] = model.predict(features)

    # Return top 5 players by prediction score
    top5 = season_data.sort_values("Prediction", ascending=False).head(5)
    return top5[["Player", "Prediction"]]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        year = request.form["year"]
        print(f"Received year: {year}")
        try:
            top5 = predict_mvp(year)
            print("Top 5 prediction successful.")
            return render_template("results.html", year=year, top5=top5.values)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template("index.html", error="Invalid year or data not available.")
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5001)