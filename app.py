"""
Premium Underwriting Assistant Flask Application with GenAI + Agentic AI
Design: Subtle, Minimal, Premium, Airy, Neo-Modern SaaS

Features:
1. Indian market standards (₹ INR currency, CIBIL scores, Indian occupations)
2. Enhanced risk scoring with multiple factors
3. GenAI chatbot integration
4. Agentic AI that can make decisions and take actions
"""

from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os
import requests
from functools import wraps

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'

# OpenRouter API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Global variables
clf = None
features = []
encoders = {}
target_le = None

# Indian market constants
INR_TO_USD = 83.0  # Current exchange rate
CIBIL_SCORE_RANGES = {
    'excellent': (750, 900),
    'good': (700, 749),
    'fair': (650, 699),
    'poor': (550, 649),
    'very_poor': (300, 549)
}

# Agentic AI Memory (stores conversation context)
AGENT_MEMORY = {}

class UnderwritingAgent:
    """
    Agentic AI - An intelligent agent that can:
    1. Analyze data
    2. Make decisions
    3. Take actions (approve/reject/request docs)
    4. Learn from context
    5. Engage in conversation
    """
    
    def __init__(self):
        self.tools = {
            'calculate_risk': self.calculate_risk,
            'check_eligibility': self.check_eligibility,
            'recommend_policy': self.recommend_policy,
            'request_documents': self.request_documents
        }
        self.memory = []
    
    def calculate_risk(self, applicant_data):
        """Tool: Calculate comprehensive risk score"""
        age = applicant_data.get('age', 30)
        income = applicant_data.get('income', 500000)
        cibil = applicant_data.get('credit_score', 700)
        claims = applicant_data.get('previous_claims', 0)
        premium = applicant_data.get('premium', 12000)
        coverage = applicant_data.get('coverage', 500000)
        occupation = applicant_data.get('occupation', 'Other')
        
        risk_score = 0
        factors = []
        
        # CIBIL Score Analysis (35% weight)
        if cibil >= 750:
            risk_score += 0
            factors.append("Excellent CIBIL score (750+)")
        elif cibil >= 700:
            risk_score += 1
            factors.append("Good CIBIL score (700-749)")
        elif cibil >= 650:
            risk_score += 2
            factors.append("Fair CIBIL score (650-699)")
        elif cibil >= 600:
            risk_score += 3.5
            factors.append("⚠️ Below average CIBIL (600-649)")
        else:
            risk_score += 5
            factors.append("⚠️ Poor CIBIL score (<600)")
        
        # Claims History (30% weight)
        if claims == 0:
            risk_score += 0
            factors.append("No previous claims - excellent record")
        elif claims == 1:
            risk_score += 2
            factors.append("1 previous claim - acceptable")
        elif claims == 2:
            risk_score += 3.5
            factors.append("⚠️ 2 previous claims - elevated risk")
        else:
            risk_score += 5
            factors.append("⚠️ Multiple claims ({}) - high risk".format(claims))
        
        # Premium Affordability (20% weight)
        annual_premium = premium * 12
        premium_ratio = annual_premium / income if income > 0 else 0
        if premium_ratio < 0.03:
            risk_score += 0
            factors.append("Premium is <3% of income - highly affordable")
        elif premium_ratio < 0.05:
            risk_score += 0.5
            factors.append("Premium is 3-5% of income - affordable")
        elif premium_ratio < 0.08:
            risk_score += 1.5
            factors.append("⚠️ Premium is 5-8% of income - moderate burden")
        else:
            risk_score += 2.5
            factors.append("⚠️ Premium >8% of income - high financial burden")
        
        # Age Analysis (10% weight)
        if 30 <= age <= 50:
            risk_score += 0
            factors.append("Prime working age (30-50)")
        elif 25 <= age < 30 or 50 < age <= 60:
            risk_score += 0.5
            factors.append("Standard age bracket")
        else:
            risk_score += 1
            factors.append("Age outside prime working years")
        
        # Occupation Risk (5% weight) - Indian context
        high_risk_occupations = ['Driver', 'Construction', 'Mining', 'Fisherman']
        medium_risk = ['Salesperson', 'Retail', 'Hospitality']
        low_risk = ['Engineer', 'Doctor', 'Teacher', 'Professional', 'Manager', 'Government Employee']
        
        if occupation in high_risk_occupations:
            risk_score += 1
            factors.append("⚠️ High-risk occupation")
        elif occupation in medium_risk:
            risk_score += 0.5
            factors.append("Medium-risk occupation")
        elif occupation in low_risk:
            risk_score += 0
            factors.append("Low-risk stable occupation")
        
        return {
            'risk_score': risk_score,
            'factors': factors,
            'classification': 'Low' if risk_score <= 3 else 'Medium' if risk_score <= 6 else 'High'
        }
    
    def check_eligibility(self, applicant_data):
        """Tool: Check if applicant meets basic eligibility"""
        age = applicant_data.get('age', 30)
        income = applicant_data.get('income', 0)
        cibil = applicant_data.get('credit_score', 700)
        
        eligibility = {
            'eligible': True,
            'reasons': []
        }
        
        # Age check
        if age < 18:
            eligibility['eligible'] = False
            eligibility['reasons'].append("Applicant must be 18+ years old")
        elif age > 70:
            eligibility['eligible'] = False
            eligibility['reasons'].append("Maximum age limit is 70 years")
        
        # Income check (minimum ₹2.4L per annum for insurance)
        if income < 240000:
            eligibility['eligible'] = False
            eligibility['reasons'].append("Minimum annual income requirement: ₹2,40,000")
        
        # CIBIL check (minimum 550 for any insurance)
        if cibil < 550:
            eligibility['eligible'] = False
            eligibility['reasons'].append("Minimum CIBIL score requirement: 550")
        
        if eligibility['eligible']:
            eligibility['reasons'].append("All basic eligibility criteria met")
        
        return eligibility
    
    def recommend_policy(self, applicant_data):
        """Tool: Recommend suitable policy based on profile"""
        income = applicant_data.get('income', 500000)
        age = applicant_data.get('age', 30)
        cibil = applicant_data.get('credit_score', 700)
        
        # Calculate ideal coverage (5-10x annual income)
        ideal_coverage = income * 7
        
        recommendations = {
            'coverage_amount': ideal_coverage,
            'premium_range': (ideal_coverage * 0.008, ideal_coverage * 0.015),
            'policy_type': '',
            'features': []
        }
        
        # Policy type recommendation
        if age < 35 and cibil >= 700:
            recommendations['policy_type'] = 'Term + Investment Plan'
            recommendations['features'] = [
                'Pure term coverage for life protection',
                'ULIP/investment component for wealth building',
                'Tax benefits under Section 80C'
            ]
        elif 35 <= age <= 50:
            recommendations['policy_type'] = 'Comprehensive Term Plan'
            recommendations['features'] = [
                'High coverage term insurance',
                'Critical illness rider',
                'Accidental death benefit'
            ]
        else:
            recommendations['policy_type'] = 'Senior Citizen Health + Life Plan'
            recommendations['features'] = [
                'Health insurance with life cover',
                'Hospitalization benefits',
                'Maturity benefits'
            ]
        
        return recommendations
    
    def request_documents(self, risk_level, applicant_data):
        """Tool: Determine required documents"""
        docs = ['PAN Card', 'Aadhaar Card', 'Income proof (last 3 months)']
        
        if risk_level == 'High':
            docs.extend([
                'Bank statements (6 months)',
                'CIBIL report',
                'Previous insurance policies',
                'Medical records',
                'Employment verification letter'
            ])
        elif risk_level == 'Medium':
            docs.extend([
                'Bank statements (3 months)',
                'Employment verification'
            ])
        
        return docs
    
    def think_and_act(self, user_query, applicant_data=None):
        """
        Agentic AI Decision Making
        The agent reasons about what to do and takes action
        """
        self.memory.append({'role': 'user', 'content': user_query})
        
        # Agent decides which tools to use based on query
        agent_prompt = f"""You are an intelligent insurance underwriting agent for the Indian market. 
You have access to these tools:
- calculate_risk: Analyze risk based on applicant data
- check_eligibility: Check if applicant meets basic criteria
- recommend_policy: Suggest suitable policy
- request_documents: List required documents

User query: {user_query}

Applicant data: {json.dumps(applicant_data) if applicant_data else 'Not provided'}

Respond directly with final professional answer only.
Do NOT reveal internal reasoning or chain of thought.
Ask clarifying questions only if necessary.
Use newline spacing and bullet points (•) for alignment."""

        response = call_genai(agent_prompt, system_prompt="You are an agentic AI that thinks, reasons, and takes action. Be analytical and precise.")
        
        self.memory.append({'role': 'agent', 'content': response})
        
        return response

# Initialize agent
underwriting_agent = UnderwritingAgent()

def call_genai(prompt, model="openai/gpt-oss-20b:free", system_prompt=None):
    headers = {
        "Authorization": f"Bearer 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",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://underwriting-ai.app",
        "X-Title": "Underwriting AI Assistant",
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    else:
        messages.append({"role": "system", "content": "You are an expert Indian underwriting assistant."})

    messages.append({"role":"user","content":prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 600
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(payload), timeout=20)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ GenAI API Error: {response.status_code} | {response.text}"
    except Exception as e:
        return f"❌ GenAI Exception occurred: {str(e)}"


def generate_ai_explanation(applicant_data, ml_prediction, ml_confidence):
    """GenAI: Natural Language Explanation (Indian Context)"""
    prompt = f"""Analyze this Indian insurance applicant and explain the risk assessment in 2-3 professional sentences:

Applicant Profile:
- Age: {applicant_data.get('age')} years
- Annual Income: ₹{applicant_data.get('income'):,}
- CIBIL Score: {applicant_data.get('credit_score')}
- Previous Claims: {applicant_data.get('previous_claims')}
- Monthly Premium: ₹{applicant_data.get('premium'):,}
- Coverage Amount: ₹{applicant_data.get('coverage'):,}
- Occupation: {applicant_data.get('occupation')}

Risk Assessment: {ml_prediction} Risk (Confidence: {ml_confidence}%)

Explain the key factors driving this risk assessment for Indian insurance standards."""

    ai_response = call_genai(prompt)
    if ai_response:
        return ai_response
    else:
        return f"The applicant presents a {ml_prediction.lower()} risk profile based on CIBIL score, claims history, and financial indicators as per Indian underwriting standards."

def load_model():
    """Load trained model"""
    global clf, features, encoders, target_le
    try:
        model_data = joblib.load('underwriting_rf_v1.pkl')
        clf = model_data['model']
        features = model_data['features']
        encoders = model_data.get('encoders', {})
        target_le = model_data.get('target_le', None)
        return True
    except Exception as e:
        print(f"Model loading error: {e}")
        return False

def decode_risk_label(enc_label):
    """Decode risk label"""
    if target_le is not None:
        try:
            return target_le.inverse_transform([int(enc_label)])[0]
        except:
            pass
    return {0: "Low", 1: "Medium", 2: "High"}.get(int(enc_label), str(enc_label))

def prepare_input_data(form_data):
    """Prepare input for ML model"""
    input_dict = {}
    
    input_dict['Age'] = int(form_data.get('age', 30))
    input_dict['Income_Level'] = float(form_data.get('income', 500000))
    input_dict['Credit_Score'] = float(form_data.get('credit_score', 700))
    input_dict['Premium_Amount'] = float(form_data.get('premium', 10000))
    input_dict['Coverage_Amount'] = float(form_data.get('coverage', 500000))
    input_dict['Deductible'] = float(form_data.get('deductible', 5000))
    input_dict['num_previous_claims'] = int(form_data.get('previous_claims', 0))
    input_dict['policy_age_days'] = 0
    input_dict['days_to_renewal'] = 365
    
    occupation = form_data.get('occupation', 'Other')
    if 'Occupation' in encoders:
        try:
            input_dict['Occupation'] = encoders['Occupation'].transform([occupation])[0]
        except:
            input_dict['Occupation'] = 0
    else:
        input_dict['Occupation'] = 0
    
    feature_vector = []
    for feat in features:
        feature_vector.append(input_dict.get(feat, 0))
    
    return np.array([feature_vector])

@app.route('/')
def index():
    """Home page"""
    session['chat_history'] = []
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Enhanced Prediction with Agentic AI"""
    try:
        form_data = request.form.to_dict()
        
        # Prepare applicant data
        applicant_data = {
            'age': int(form_data.get('age', 30)),
            'income': float(form_data.get('income', 500000)),
            'credit_score': float(form_data.get('credit_score', 700)),
            'previous_claims': int(form_data.get('previous_claims', 0)),
            'premium': float(form_data.get('premium', 10000)),
            'coverage': float(form_data.get('coverage', 500000)),
            'occupation': form_data.get('occupation', 'Other')
        }
        
        # 1. Use Agentic AI to calculate comprehensive risk
        agent_risk_analysis = underwriting_agent.calculate_risk(applicant_data)
        
        # 2. Check eligibility
        eligibility = underwriting_agent.check_eligibility(applicant_data)
        
        # 3. ML Model prediction
        X = prepare_input_data(form_data)
        pred_class = clf.predict(X)[0]
        pred_proba = clf.predict_proba(X)[0]
        pred_label = decode_risk_label(pred_class)
        confidence = float(pred_proba[pred_class]) * 100
        
        # 4. GenAI explanation
        ai_explanation = generate_ai_explanation(applicant_data, pred_label, round(confidence, 2))
        
        # 5. Get policy recommendations
        policy_recommendations = underwriting_agent.recommend_policy(applicant_data)
        
        # 6. Required documents
        required_docs = underwriting_agent.request_documents(agent_risk_analysis['classification'], applicant_data)
        
        # Calculate risk score (0-100)
        risk_score = int((agent_risk_analysis['risk_score'] / 12) * 100)
        
        result = {
            'success': True,
            'risk_level': agent_risk_analysis['classification'],
            'confidence': round(confidence, 2),
            'risk_score': risk_score,
            'ai_explanation': ai_explanation,
            'risk_factors': agent_risk_analysis['factors'],
            'eligibility': eligibility,
            'policy_recommendations': policy_recommendations,
            'required_documents': required_docs,
            'probabilities': {
                'low': round(float(pred_proba[0]) * 100, 2),
                'medium': round(float(pred_proba[1]) * 100, 2) if len(pred_proba) > 1 else 0,
                'high': round(float(pred_proba[2]) * 100, 2) if len(pred_proba) > 2 else 0
            },
            'genai_enabled': True
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/chat', methods=['POST'])
def chat():
    """
    Agentic AI Chatbot Endpoint
    User can ask questions and the agent responds with actions
    """
    try:
        data = request.json
        user_message = data.get('message', '')
        applicant_data = data.get('applicant_data', None)
        
        # Get chat history from session
        chat_history = session.get('chat_history', [])
        
        # Agent thinks and responds
        agent_response = underwriting_agent.think_and_act(user_message, applicant_data)
        
        # Update chat history
        chat_history.append({'role': 'user', 'message': user_message})
        chat_history.append({'role': 'agent', 'message': agent_response})
        session['chat_history'] = chat_history[-10:]  # Keep last 10 messages
        
        return jsonify({
            'success': True,
            'response': agent_response,
            'chat_history': chat_history
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def create_dummy_model():
    """Create realistic model for Indian market"""
    global clf, features
    
    print("Creating enhanced model for Indian insurance market...")
    
    features = [
        'Age', 'Income_Level', 'Credit_Score', 'Premium_Amount',
        'Coverage_Amount', 'Deductible', 'num_previous_claims',
        'policy_age_days', 'days_to_renewal', 'Occupation'
    ]
    
    np.random.seed(42)
    n_samples = 1500
    
    # Indian market distributions (in INR)
    ages = np.random.normal(38, 13, n_samples).clip(22, 68).astype(int)
    incomes = np.random.lognormal(13.2, 0.6, n_samples).clip(240000, 5000000)  # ₹2.4L to ₹50L
    cibil_scores = np.random.normal(690, 85, n_samples).clip(350, 900)  # CIBIL range
    premiums = np.random.lognormal(9.2, 0.7, n_samples).clip(3000, 50000)  # ₹3K to ₹50K monthly
    coverages = np.random.lognormal(13.8, 0.7, n_samples).clip(500000, 10000000)  # ₹5L to ₹1Cr
    deductibles = np.random.choice([5000, 10000, 25000, 50000], n_samples)
    
    # Claims distribution
    claims_probs = [0.52, 0.28, 0.12, 0.05, 0.03]
    claims = np.random.choice([0, 1, 2, 3, 4], n_samples, p=claims_probs)
    
    policy_ages = np.random.randint(0, 1095, n_samples)
    days_renewal = np.random.randint(0, 365, n_samples)
    occupations = np.random.randint(0, 10, n_samples)
    
    X_dummy = np.column_stack([
        ages, incomes, cibil_scores, premiums, coverages,
        deductibles, claims, policy_ages, days_renewal, occupations
    ])
    
    # Enhanced risk calculation
    y_dummy = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        temp_data = {
            'age': X_dummy[i, 0],
            'income': X_dummy[i, 1],
            'credit_score': X_dummy[i, 2],
            'premium': X_dummy[i, 3],
            'coverage': X_dummy[i, 4],
            'previous_claims': X_dummy[i, 6],
            'occupation': 'Professional'
        }
        risk_result = underwriting_agent.calculate_risk(temp_data)
        y_dummy[i] = 0 if risk_result['classification'] == 'Low' else 1 if risk_result['classification'] == 'Medium' else 2
    
    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=18,
        min_samples_split=15,
        min_samples_leaf=8,
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(X_dummy, y_dummy)
    
    unique, counts = np.unique(y_dummy, return_counts=True)
    distribution = dict(zip(unique, counts))
    
    print(f"✓ Model created for Indian market")
    print(f"  - Training samples: {n_samples}")
    print(f"  - Risk distribution:")
    print(f"    • Low Risk:    {distribution.get(0, 0):4d} ({distribution.get(0, 0)/n_samples*100:.1f}%)")
    print(f"    • Medium Risk: {distribution.get(1, 0):4d} ({distribution.get(1, 0)/n_samples*100:.1f}%)")
    print(f"    • High Risk:   {distribution.get(2, 0):4d} ({distribution.get(2, 0)/n_samples*100:.1f}%)")

if __name__ == '__main__':
    print("="*70)
    print("Premium Underwriting AI Assistant - Indian Market Edition")
    print("="*70)
    
    model_loaded = load_model()
    
    if not model_loaded:
        print("\n⚠ Warning: Saved model not found.")
        print("  Creating model with Indian market standards...\n")
        create_dummy_model()
    else:
        print("\n✓ Model loaded successfully")
    
    print("\n🤖 AI Features Enabled:")
    print("  ✓ Agentic AI with reasoning & decision-making")
    print("  ✓ GPT-4 powered chatbot")
    print("  ✓ Indian market standards (INR, CIBIL, Indian occupations)")
    print("  ✓ Multi-factor risk scoring (CIBIL, claims, income, age, occupation)")
    
    print("\n" + "="*70)
    print("Starting Flask server...")
    print("Visit: http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)