from flask import Flask, render_template, request, redirect, session, flash
import sqlite3
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import re
import pickle
import random
import nltk
import json

# NLTK setup
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir=nltk_data_dir)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'secret_key_123')

# Create directories
UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
GRAPH_FOLDER = os.path.join("static", "graphs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)

# ---------- SQLITE DATABASE ----------
DB_PATH = os.path.join(os.path.dirname(__file__), 'hate_speech.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    
    # Users table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        email TEXT,
        mobile TEXT,
        address TEXT,
        status TEXT DEFAULT 'active',
        warning_count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # History table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        text TEXT,
        prediction TEXT,
        admin_action TEXT DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Admin table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS admin(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        password TEXT
    )
    ''')
    
    # Insert default admin
    cur.execute("SELECT * FROM admin WHERE username='admin'")
    if not cur.fetchone():
        cur.execute("INSERT INTO admin(username,password) VALUES('admin','admin')")
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")

# Initialize database
init_db()

# ---------- HELPER FUNCTIONS ----------
def admin_required():
    return "admin" in session

def user_required():
    return "user" in session

def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [stemmer.stem(w) for w in words]
    
    return " ".join(words)

def get_dataset_path():
    return session.get("dataset_path")

def get_preprocessed_path():
    return session.get("preprocessed_path")

def load_saved_vectorizer():
    vectorizer_path = os.path.join(MODEL_FOLDER, "vectorizer.pkl")
    if os.path.exists(vectorizer_path):
        with open(vectorizer_path, "rb") as f:
            return pickle.load(f)
    return None

def load_saved_best_model():
    best_model_path = os.path.join(MODEL_FOLDER, "best_model.pkl")
    if os.path.exists(best_model_path):
        with open(best_model_path, "rb") as f:
            return pickle.load(f)
    return None

def load_best_model_name():
    path = os.path.join(MODEL_FOLDER, "best_model_name.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "Not Available"

# ---------- ROUTES ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()
        email = request.form["email"].strip()
        mobile = request.form["mobile"].strip()
        address = request.form["address"].strip()

        conn = get_db()
        cur = conn.cursor()

        cur.execute("SELECT * FROM users WHERE username=?", (username,))
        existing = cur.fetchone()

        if existing:
            conn.close()
            flash("Username already exists")
            return redirect("/register")

        cur.execute(
            "INSERT INTO users(username,password,email,mobile,address,status,warning_count) VALUES(?,?,?,?,?,?,?)",
            (username, password, email, mobile, address, "active", 0)
        )
        conn.commit()
        conn.close()

        flash("Registration successful")
        return redirect("/login")

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()

        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        )
        data = cur.fetchone()
        conn.close()

        if data:
            if data["status"] == "blocked":
                flash("Your account is blocked by admin")
                return redirect("/login")

            session["user"] = username
            flash("Login successful")
            return redirect("/predict")
        else:
            flash("Invalid username or password")

    return render_template("login.html")

@app.route("/admin", methods=["GET", "POST"])
def admin():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()

        if username == "admin" and password == "admin":
            session["admin"] = "admin"
            flash("Admin login successful")
            return redirect("/admin_home")
        else:
            flash("Invalid admin credentials")

    return render_template("admin_login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully")
    return redirect("/")

@app.route("/admin_home")
def admin_home():
    if not admin_required():
        return redirect("/admin")
    return render_template("admin_home.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if not admin_required():
        return redirect("/admin")

    preview_data = []
    columns = []

    if request.method == "POST":
        file = request.files.get("dataset")

        if not file or file.filename == "":
            flash("Please choose a CSV dataset file")
            return redirect("/upload")

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)
        session["dataset_path"] = path

        try:
            df = pd.read_csv(path)

            if "text" not in df.columns or "label" not in df.columns:
                flash("Dataset must contain 'text' and 'label' columns")
                return redirect("/upload")

            sample_size = min(10, len(df))
            preview_df = df.sample(n=sample_size, random_state=random.randint(1, 10000))
            preview_data = preview_df.to_dict(orient="records")
            columns = list(df.columns)

            flash("Dataset uploaded successfully")
        except Exception as e:
            flash(f"Error reading dataset: {str(e)}")
            return redirect("/upload")

    return render_template("upload.html", preview_data=preview_data, columns=columns)

@app.route("/preprocess")
def preprocess():
    if not admin_required():
        return redirect("/admin")

    path = get_dataset_path()
    if not path or not os.path.exists(path):
        flash("Please upload dataset first")
        return redirect("/upload")

    try:
        df = pd.read_csv(path)

        if "text" not in df.columns or "label" not in df.columns:
            flash("Dataset must contain 'text' and 'label' columns")
            return redirect("/upload")

        before_rows = len(df)

        df = df[["text", "label"]].copy()
        df.dropna(subset=["text", "label"], inplace=True)
        df["text"] = df["text"].apply(clean_text)
        df["label"] = df["label"].astype(str).str.lower().str.strip()
        df = df[df["text"] != ""]
        df.drop_duplicates(inplace=True)

        after_rows = len(df)

        preprocessed_path = os.path.join(UPLOAD_FOLDER, "preprocessed_dataset.csv")
        df.to_csv(preprocessed_path, index=False)
        session["preprocessed_path"] = preprocessed_path

        preview_df = df.head(10)
        preview_data = preview_df.to_dict(orient="records")
        columns = list(df.columns)

        flash("Dataset preprocessing with NLTK completed successfully")

        return render_template(
            "preprocess.html",
            preview_data=preview_data,
            columns=columns,
            before_rows=before_rows,
            after_rows=after_rows,
            removed_rows=before_rows - after_rows
        )

    except Exception as e:
        flash(f"Preprocess error: {str(e)}")
        return redirect("/upload")

@app.route("/train")
def train():
    if not admin_required():
        return redirect("/admin")

    path = get_preprocessed_path()
    if not path or not os.path.exists(path):
        flash("Please preprocess dataset first")
        return redirect("/preprocess")

    try:
        df = pd.read_csv(path)

        if "text" not in df.columns or "label" not in df.columns:
            flash("Preprocessed dataset must contain 'text' and 'label'")
            return redirect("/preprocess")

        X = df["text"].astype(str)
        y = df["label"].astype(str)

        cv = CountVectorizer()
        X_vec = cv.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_vec, y, test_size=0.2, random_state=42, stratify=y
        )

        models = {
            "Naive Bayes": MultinomialNB(),
            "SVM": LinearSVC(),
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }

        scores = {}
        best_model_name = None
        best_accuracy = 0.0
        best_model = None
        label_order = sorted(list(y.unique()))

        with open(os.path.join(MODEL_FOLDER, "vectorizer.pkl"), "wb") as f:
            pickle.dump(cv, f)

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            scores[name] = round(acc * 100, 2)

            model_file = name.lower().replace(" ", "_") + ".pkl"
            with open(os.path.join(MODEL_FOLDER, model_file), "wb") as f:
                pickle.dump(model, f)

            cm = confusion_matrix(y_test, y_pred, labels=label_order)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_order)
            fig, ax = plt.subplots(figsize=(6, 6))
            disp.plot(ax=ax, cmap="Blues", colorbar=False)
            plt.title(f"{name} Confusion Matrix")
            plt.tight_layout()

            image_name = name.lower().replace(" ", "_") + "_cm.png"
            plt.savefig(os.path.join(GRAPH_FOLDER, image_name))
            plt.close()

            if acc > best_accuracy:
                best_accuracy = acc
                best_model_name = name
                best_model = model

        with open(os.path.join(MODEL_FOLDER, "best_model.pkl"), "wb") as f:
            pickle.dump(best_model, f)

        with open(os.path.join(MODEL_FOLDER, "best_model_name.txt"), "w", encoding="utf-8") as f:
            f.write(best_model_name)

        plt.figure(figsize=(8, 5))
        plt.bar(scores.keys(), scores.values())
        plt.title("AI Algorithm Accuracy Comparison")
        plt.xlabel("Algorithms")
        plt.ylabel("Accuracy (%)")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPH_FOLDER, "comparison.png"))
        plt.close()

        flash("AI model training completed and models saved successfully")

        return render_template(
            "train.html",
            scores=scores,
            best_model_name=best_model_name,
            best_accuracy=round(best_accuracy * 100, 2),
            comparison_graph="graphs/comparison.png",
            nb_cm="graphs/naive_bayes_cm.png",
            svm_cm="graphs/svm_cm.png",
            lr_cm="graphs/logistic_regression_cm.png",
            rf_cm="graphs/random_forest_cm.png"
        )

    except Exception as e:
        flash(f"Training error: {str(e)}")
        return redirect("/admin_home")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not user_required():
        return redirect("/login")

    result = None
    text = ""
    model_name = load_best_model_name()

    if request.method == "POST":
        text = request.form["text"].strip()

        vectorizer = load_saved_vectorizer()
        model = load_saved_best_model()

        if vectorizer is None or model is None:
            flash("Model is not trained yet. Please contact admin.")
            return redirect("/predict")

        try:
            clean_input = clean_text(text)
            input_vec = vectorizer.transform([clean_input])
            pred = model.predict(input_vec)[0]
            result = pred

            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO history(username,text,prediction,admin_action) VALUES(?,?,?,?)",
                (session["user"], text, pred, "pending")
            )
            conn.commit()
            conn.close()

            flash("AI prediction completed successfully")

        except Exception as e:
            flash(f"Prediction error: {str(e)}")

    return render_template("predict.html", result=result, text=text, model_name=model_name)

@app.route("/history")
def history():
    if not user_required():
        return redirect("/login")

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, username, text, prediction, admin_action
        FROM history
        WHERE username=?
        ORDER BY id DESC
    """, (session["user"],))
    data = cur.fetchall()
    conn.close()

    return render_template("history.html", data=data)

@app.route("/users")
def users():
    if not admin_required():
        return redirect("/admin")

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users ORDER BY id DESC")
    data = cur.fetchall()
    conn.close()

    return render_template("users.html", data=data)

@app.route("/all_history")
def all_history():
    if not admin_required():
        return redirect("/admin")

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM history ORDER BY id DESC")
    data = cur.fetchall()
    conn.close()

    return render_template("all_history.html", data=data)

@app.route("/review_predictions")
def review_predictions():
    if not admin_required():
        return redirect("/admin")

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT h.id, h.username, h.text, h.prediction, h.admin_action,
               u.status, u.warning_count
        FROM history h
        JOIN users u ON h.username = u.username
        ORDER BY h.id DESC
    """)
    data = cur.fetchall()
    conn.close()

    return render_template("review_predictions.html", data=data)

@app.route("/give_warning/<int:history_id>/<username>")
def give_warning(history_id, username):
    if not admin_required():
        return redirect("/admin")

    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT * FROM history WHERE id=?", (history_id,))
    row = cur.fetchone()

    if row and row["prediction"] == "hate" and row["admin_action"] == "pending":
        cur.execute(
            "UPDATE users SET warning_count = warning_count + 1 WHERE username=?",
            (username,)
        )
        cur.execute(
            "UPDATE history SET admin_action=? WHERE id=?",
            ("warned", history_id)
        )
        conn.commit()
        flash(f"Warning given to {username}")
    else:
        flash("Warning action not allowed")

    conn.close()
    return redirect("/review_predictions")

@app.route("/block_user/<int:history_id>/<username>")
def block_user(history_id, username):
    if not admin_required():
        return redirect("/admin")

    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT * FROM history WHERE id=?", (history_id,))
    row = cur.fetchone()

    if row and row["prediction"] == "offensive" and row["admin_action"] == "pending":
        cur.execute(
            "UPDATE users SET status=? WHERE username=?",
            ("blocked", username)
        )
        cur.execute(
            "UPDATE history SET admin_action=? WHERE id=?",
            ("blocked", history_id)
        )
        conn.commit()
        flash(f"User {username} has been blocked")
    else:
        flash("Block action not allowed")

    conn.close()
    return redirect("/review_predictions")

@app.route("/unblock_user/<username>")
def unblock_user(username):
    if not admin_required():
        return redirect("/admin")

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET status=? WHERE username=?",
        ("active", username)
    )
    conn.commit()
    conn.close()

    flash(f"User {username} unblocked successfully")
    return redirect("/users")

@app.route("/delete_history/<int:id>")
def delete_history(id):
    if not user_required():
        return redirect("/login")

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM history WHERE id=? AND username=?",
        (id, session["user"])
    )
    conn.commit()
    conn.close()

    flash("History deleted successfully")
    return redirect("/history")

@app.route("/admin_delete_history/<int:id>")
def admin_delete_history(id):
    if not admin_required():
        return redirect("/admin")

    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM history WHERE id=?", (id,))
    conn.commit()
    conn.close()

    flash("History deleted by admin")
    return redirect("/all_history")

@app.route("/admin_delete_all_history")
def admin_delete_all_history():
    if not admin_required():
        return redirect("/admin")

    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM history")
    conn.commit()
    conn.close()

    flash("All history deleted by admin")
    return redirect("/all_history")

# ---------- ERROR HANDLERS ----------
@app.errorhandler(404)
def not_found(e):
    return render_template("index.html"), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template("index.html", error="Internal server error"), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
