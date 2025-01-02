import os
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random

app = Flask(__name__)

# Initialize the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Timer settings for each question (in seconds)
QUESTION_TIMER = 60

# Global variables for tracking score and progress
score = 0
total_questions = 5
questions = []
current_question_index = 0
total_similarity_score = 0  # Track total similarity score

# Function to calculate semantic similarity between user and correct answers
def calculate_semantic_similarity(user_answer, correct_answer):
    embeddings = model.encode([user_answer, correct_answer])
    cosine_sim = cosine_similarity([embeddings[0]], [embeddings[1]])
    return cosine_sim[0][0]

# Function to load the dataset based on the domain
def get_domain_dataset(domain, uploaded_files):
    domain_normalized = domain.replace(" ", "_").lower()
    for filename in uploaded_files:
        filename_normalized = filename.replace(" ", "_").lower()
        if domain_normalized in filename_normalized:
            try:
                return pd.read_csv(os.path.join(os.getcwd(), filename))
            except Exception as e:
                print(f"Error loading file {filename}: {e}")
                return None
    return None

@app.route('/')
def index():
    domain_options = [
        "Software Development", "Data_Science", "Machine Learning", "Artificial_Intelligence",
        "Data_Engineer", "Business Intelligence", "Cloud Computing", "Cybersecurity",
        "DevOps", "Networking", "Database Administration", "System Administration",
        "Full Stack Development", "Front-End Development", "Back-End Development",
        "Quality Assurance (QA)", "Game Development", "Mobile App Development",
        "UX UI_Design", "Product Management", "Project Management", "Finance or Quantitative Analysis",
        "Digital Marketing", "Human Resources", "Sales", "Customer Support",
        "Operations Management", "Healthcare", "Research and Development", "Consulting",
        "Legal and Compliance", "Education and Training", "Blockchain", "Internet of Things (IoT)",
        "Robotics", "Ethical Hacking and Penetration Testing", "Business Analysis", "Accounting",
        "Product Design", "Supply Chain Management", "Public Relations", "Technical Writing",
        "Social Media Management", "Sustainability and Environmental Management",
        "Hospitality and Event Management", "Vlsi", "Embedded", "Cloud Architect",
        "Chartered Accountant", "Excel Expert", "Chemical Engineering", "Quantum Computing","Chemistry"
    ]
    domain_options.sort()
    return render_template('index.html', domain_options=domain_options)



@app.route('/start_interview', methods=['POST'])
def start_interview():
    global questions, score, current_question_index, total_similarity_score
    score = 0
    current_question_index = 0
    total_similarity_score = 0

    user_name = request.args.get('name')
    selected_domain = request.form['domain']
    uploaded_files = os.listdir(os.getcwd())  # List files in the current directory
    dataset = get_domain_dataset(selected_domain, uploaded_files)
    
    if dataset is not None:
        if {'Domain', 'Question', 'Answer'}.issubset(dataset.columns):
            dataset_cleaned = dataset.dropna(subset=['Question', 'Answer'])
            selected_dataset = dataset_cleaned[['Domain', 'Question', 'Answer']]
            
            # Select only 5 random questions
            questions = selected_dataset.sample(n=5, replace=False).to_dict(orient='records')
            return render_template('interview.html', question=questions[0], question_index=current_question_index, timer=QUESTION_TIMER, name=user_name)
        else:
            return "Dataset missing required columns."
    else:
        return "Dataset not found for the selected domain."

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    global score, current_question_index, total_similarity_score
    try:
        user_answer = request.form.get('user_answer', None)
        question_index = int(request.form['question_index'])
        correct_answer = questions[question_index]['Answer']

        # Handle skipping (no answer submitted)
        if user_answer == "":
            result = "Skipped"
            similarity = 0.0  # No similarity score if skipped
        else:
            # Calculate similarity only if an answer is provided
            similarity = float(calculate_semantic_similarity(user_answer, correct_answer))
            if similarity >= 0.5:
                score += 1
                result = "Correct!"
            else:
                result = "Incorrect."

        total_similarity_score += similarity

        # Move to the next question or finish the interview
        current_question_index += 1
        if current_question_index < total_questions:
            next_question = questions[current_question_index]
            return jsonify({
                'similarity': round(similarity, 2),
                'result': result,
                'next_question': next_question['Question'],
                'question_index': current_question_index,
                'timer': QUESTION_TIMER
            })
        else:
            # Calculate total similarity score for the entire interview
            similarity_score = total_similarity_score / total_questions
            return jsonify({
                'similarity': round(similarity_score, 2),
                'result': result,
                'finished': True,
                'score': score,
                'total': total_questions,
                'total_similarity_score': round(similarity_score, 2)
            })
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/show_scores')
def show_scores():
    user_name = request.args.get('name')
    final_score = request.args.get('final_score', type=int)
    average_similarity_score = request.args.get('average_similarity_score', type=float)
    return render_template('scores.html', user_name=user_name, final_score=final_score, average_similarity_score=average_similarity_score)

if __name__ == '__main__':
    app.run(debug=True)

#if __name__ == '__main__':
#   app.run(host='0.0.0.0', port=7860)
