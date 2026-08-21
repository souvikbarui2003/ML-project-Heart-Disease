import { Heart, Shield, Activity, BarChart3, AlertTriangle, Brain, BookOpen } from 'lucide-react'
import type { Page } from '../App'

interface Props {
  onNavigate: (page: Page) => void
}

export default function LandingPage({ onNavigate }: Props) {
  return (
    <div className="space-y-16">
      {/* Hero Section */}
      <section className="text-center py-12 sm:py-20">
        <div className="inline-flex items-center gap-2 bg-red-50 text-red-700 px-4 py-1.5 rounded-full text-sm font-medium mb-6">
          <Heart className="w-4 h-4 fill-red-500" />
          Machine Learning Risk Assessment
        </div>
        <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-gray-900 leading-tight mb-6">
          Heart Disease Risk
          <br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-600 to-red-500">
            Prediction Engine
          </span>
        </h1>
        <p className="text-lg sm:text-xl text-gray-600 max-w-2xl mx-auto mb-10 leading-relaxed">
          A machine-learning tool that estimates heart-disease risk based on clinical features
          learned from the UCI Heart Disease dataset. Built with scikit-learn, XGBoost, and SHAP
          for explainable predictions.
        </p>
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <button
            onClick={() => onNavigate('predict')}
            className="btn-primary text-lg px-8 py-4"
          >
            <Activity className="w-5 h-5 inline mr-2" />
            Try Prediction
          </button>
          <button
            onClick={() => onNavigate('dashboard')}
            className="btn-secondary text-lg px-8 py-4"
          >
            <BarChart3 className="w-5 h-5 inline mr-2" />
            View Dashboard
          </button>
          <button
            onClick={() => onNavigate('research')}
            className="btn-secondary text-lg px-8 py-4"
          >
            <BookOpen className="w-5 h-5 inline mr-2" />
            Research Paper
          </button>
        </div>
      </section>

      {/* Disclaimer Banner */}
      <section className="bg-amber-50 border border-amber-200 rounded-2xl p-6">
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-6 h-6 text-amber-600 mt-0.5 flex-shrink-0" />
          <div>
            <h3 className="font-semibold text-amber-800 mb-1">Important Disclaimer</h3>
            <p className="text-sm text-amber-700 leading-relaxed">
              This is a <strong>research and educational tool</strong>, not a medical device.
              Model-estimated risk scores are based on statistical patterns in a limited dataset
              and should <strong>never</strong> replace professional medical evaluation. Always
              consult a qualified healthcare professional for diagnosis and treatment decisions.
            </p>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section>
        <h2 className="text-2xl font-bold text-center text-gray-900 mb-10">How It Works</h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
          <FeatureCard
            icon={<Brain className="w-6 h-6 text-primary-600" />}
            title="Trained on Clinical Data"
            description="The model was trained on the UCI Cleveland Heart Disease dataset (303 patients, 13 clinical features) using stratified cross-validation."
          />
          <FeatureCard
            icon={<Activity className="w-6 h-6 text-primary-600" />}
            title="Multiple Algorithms Compared"
            description="Six algorithms were evaluated: Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, and SVM. The best model was selected by CV ROC-AUC."
          />
          <FeatureCard
            icon={<Shield className="w-6 h-6 text-primary-600" />}
            title="Explainable Predictions"
            description="Feature importance analysis shows which clinical variables contributed most to each prediction. No black-box decisions."
          />
          <FeatureCard
            icon={<BarChart3 className="w-6 h-6 text-primary-600" />}
            title="Probability-Based Risk"
            description="Outputs a continuous probability score (0–1) and a risk category, rather than a simple yes/no answer."
          />
          <FeatureCard
            icon={<Heart className="w-6 h-6 text-primary-600" />}
            title="Healthcare-Focused Metrics"
            description="Evaluation prioritizes recall (sensitivity) and specificity — critical for health screening where missing a case is costly."
          />
          <FeatureCard
            icon={<Shield className="w-6 h-6 text-primary-600" />}
            title="Reproducible Pipeline"
            description="Fixed random seeds, proper train/test splitting before preprocessing, and no data leakage. Full results reproducible with one command."
          />
        </div>
      </section>

      {/* Dataset Info */}
      <section className="card">
        <h2 className="text-xl font-bold text-gray-900 mb-4">About the Dataset</h2>
        <div className="grid sm:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold text-gray-700 mb-2">Source</h3>
            <p className="text-sm text-gray-600 leading-relaxed">
              UCI Machine Learning Repository — Cleveland Heart Disease dataset.
              Originally collected by the Cleveland Clinic Foundation.
            </p>
            <h3 className="font-semibold text-gray-700 mb-2 mt-4">Size</h3>
            <p className="text-sm text-gray-600">303 patients, 13 clinical features, 1 binary target</p>
          </div>
          <div>
            <h3 className="font-semibold text-gray-700 mb-2">Features Include</h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Age, sex, chest pain type</li>
              <li>• Resting blood pressure, cholesterol</li>
              <li>• Fasting blood sugar, resting ECG</li>
              <li>• Maximum heart rate, exercise angina</li>
              <li>• ST depression, slope, vessels, thalassemia</li>
            </ul>
            <h3 className="font-semibold text-gray-700 mb-2 mt-4">Limitations</h3>
            <p className="text-sm text-gray-600">
              Small sample size (303), limited demographic diversity, single-clinic origin.
              Results may not generalize to all populations.
            </p>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="text-center py-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Ready to Try It?</h2>
        <p className="text-gray-600 mb-6 max-w-lg mx-auto">
          Enter clinical features and see the model's risk estimation with an explanation of the
          key contributing factors.
        </p>
        <button onClick={() => onNavigate('predict')} className="btn-primary text-lg px-8 py-4">
          <Activity className="w-5 h-5 inline mr-2" />
          Start Prediction
        </button>
      </section>
    </div>
  )
}

function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode
  title: string
  description: string
}) {
  return (
    <div className="card card-hover">
      <div className="w-12 h-12 bg-primary-50 rounded-xl flex items-center justify-center mb-4">
        {icon}
      </div>
      <h3 className="font-semibold text-gray-900 mb-2">{title}</h3>
      <p className="text-sm text-gray-600 leading-relaxed">{description}</p>
    </div>
  )
}
