import { Heart, Shield, Activity, BarChart3, AlertTriangle, Brain, BookOpen, Sparkles, Zap, Target } from 'lucide-react'
import FloatingHearts from '../components/FloatingHearts'
import type { Page } from '../App'

interface Props {
  onNavigate: (page: Page) => void
}

export default function LandingPage({ onNavigate }: Props) {
  return (
    <div className="space-y-12 sm:space-y-16">
      {/* Hero Section */}
      <section className="relative text-center py-8 sm:py-12 lg:py-20 overflow-hidden">
        {/* Background decoration */}
        <div className="absolute inset-0 -z-10">
          <div className="absolute top-10 left-10 w-40 h-40 bg-rose-200/30 rounded-full blur-3xl" />
          <div className="absolute bottom-10 right-10 w-56 h-56 bg-orange-200/30 rounded-full blur-3xl" />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-80 h-80 bg-pink-100/20 rounded-full blur-3xl" />
        </div>

        <FloatingHearts count={15} />

        {/* Badge */}
        <div className="inline-flex items-center gap-2 bg-gradient-to-r from-rose-50 to-orange-50 border border-rose-200 text-rose-700 px-4 py-1.5 rounded-full text-xs sm:text-sm font-medium mb-6 shadow-sm">
          <Heart className="w-3.5 h-3.5 sm:w-4 sm:h-4 fill-rose-500 animate-pulse-heart" />
          Machine Learning Risk Assessment
        </div>

        {/* Heart Illustration */}
        <div className="flex justify-center mb-6">
          <div className="relative float-3d">
            <img src="/heart-illustration.svg" alt="" className="w-24 h-24 sm:w-32 sm:h-32 lg:w-40 lg:h-40 drop-shadow-xl" />
            <div className="absolute -top-2 -right-2 w-8 h-8 bg-gradient-to-br from-yellow-400 to-orange-500 rounded-full flex items-center justify-center shadow-lg animate-bounce">
              <Sparkles className="w-4 h-4 text-white" />
            </div>
          </div>
        </div>

        {/* Title */}
        <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-extrabold leading-tight mb-4 sm:mb-6">
          <span className="text-gray-900">Heart Disease Risk</span>
          <br />
          <span className="text-gradient-heart">Prediction Engine</span>
        </h1>

        <p className="text-base sm:text-lg lg:text-xl text-gray-600 max-w-2xl mx-auto mb-6 sm:mb-10 leading-relaxed px-2">
          A machine-learning tool that estimates heart-disease risk based on clinical features
          learned from the UCI Heart Disease dataset. Built with scikit-learn and explainable AI.
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-3 sm:gap-4">
          <button
            onClick={() => onNavigate('predict')}
            className="btn-primary w-full sm:w-auto group"
          >
            <Activity className="w-4 h-4 sm:w-5 sm:h-5 inline mr-2 group-hover:animate-bounce" />
            Try Prediction
          </button>
          <button
            onClick={() => onNavigate('dashboard')}
            className="btn-secondary w-full sm:w-auto group"
          >
            <BarChart3 className="w-4 h-4 sm:w-5 sm:h-5 inline mr-2 group-hover:rotate-12 transition-transform" />
            View Dashboard
          </button>
          <button
            onClick={() => onNavigate('research')}
            className="btn-secondary w-full sm:w-auto group"
          >
            <BookOpen className="w-4 h-4 sm:w-5 sm:h-5 inline mr-2 group-hover:-rotate-12 transition-transform" />
            Research Paper
          </button>
        </div>
      </section>

      {/* Disclaimer Banner */}
      <section className="relative overflow-hidden bg-gradient-to-r from-amber-50 via-orange-50 to-rose-50 border border-amber-200 rounded-2xl p-4 sm:p-6 shadow-sm">
        <div className="absolute top-0 right-0 w-20 h-20 bg-amber-200/30 rounded-full -mr-10 -mt-10 blur-xl" />
        <div className="flex items-start gap-3 relative">
          <div className="w-10 h-10 bg-gradient-to-br from-amber-400 to-orange-500 rounded-xl flex items-center justify-center shrink-0 shadow-md">
            <AlertTriangle className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="font-bold text-amber-800 mb-1 text-sm sm:text-base">Important Disclaimer</h3>
            <p className="text-xs sm:text-sm text-amber-700 leading-relaxed">
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
        <div className="text-center mb-8 sm:mb-10">
          <h2 className="text-2xl sm:text-3xl font-extrabold text-gray-900 mb-3">How It Works</h2>
          <div className="w-20 h-1 bg-gradient-to-r from-rose-400 to-orange-400 rounded-full mx-auto" />
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
          <FeatureCard
            icon={<Brain className="w-6 h-6" />}
            title="Trained on Clinical Data"
            description="The model was trained on the UCI Cleveland Heart Disease dataset (303 patients, 13 clinical features) using stratified cross-validation."
            gradient="from-blue-500 to-indigo-600"
            bgGradient="from-blue-50 to-indigo-50"
          />
          <FeatureCard
            icon={<Activity className="w-6 h-6" />}
            title="Multiple Algorithms Compared"
            description="Six algorithms were evaluated: Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, and SVM."
            gradient="from-rose-500 to-pink-600"
            bgGradient="from-rose-50 to-pink-50"
          />
          <FeatureCard
            icon={<Shield className="w-6 h-6" />}
            title="Explainable Predictions"
            description="Feature importance analysis shows which clinical variables contributed most to each prediction. No black-box decisions."
            gradient="from-emerald-500 to-teal-600"
            bgGradient="from-emerald-50 to-teal-50"
          />
          <FeatureCard
            icon={<BarChart3 className="w-6 h-6" />}
            title="Probability-Based Risk"
            description="Outputs a continuous probability score (0–1) and a risk category, rather than a simple yes/no answer."
            gradient="from-orange-500 to-red-600"
            bgGradient="from-orange-50 to-red-50"
          />
          <FeatureCard
            icon={<Target className="w-6 h-6" />}
            title="Healthcare-Focused Metrics"
            description="Evaluation prioritizes recall (sensitivity) and specificity — critical for health screening where missing a case is costly."
            gradient="from-purple-500 to-violet-600"
            bgGradient="from-purple-50 to-violet-50"
          />
          <FeatureCard
            icon={<Zap className="w-6 h-6" />}
            title="Reproducible Pipeline"
            description="Fixed random seeds, proper train/test splitting before preprocessing, and no data leakage. Full results reproducible."
            gradient="from-amber-500 to-orange-600"
            bgGradient="from-amber-50 to-orange-50"
          />
        </div>
      </section>

      {/* Dataset Info */}
      <section className="relative overflow-hidden">
        <div className="absolute -top-10 -right-10 w-40 h-40 bg-rose-100/40 rounded-full blur-3xl" />
        <div className="card-3d relative">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-gradient-to-br from-rose-400 to-red-500 rounded-xl flex items-center justify-center shadow-md">
              <Heart className="w-5 h-5 text-white fill-white" />
            </div>
            <h2 className="text-xl font-bold text-gray-900">About the Dataset</h2>
          </div>
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
                {['Age, sex, chest pain type', 'Resting blood pressure, cholesterol', 'Fasting blood sugar, resting ECG', 'Maximum heart rate, exercise angina', 'ST depression, slope, vessels, thalassemia'].map((item, i) => (
                  <li key={i} className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-gradient-to-r from-rose-400 to-orange-400 shrink-0" />
                    {item}
                  </li>
                ))}
              </ul>
              <h3 className="font-semibold text-gray-700 mb-2 mt-4">Limitations</h3>
              <p className="text-sm text-gray-600">
                Small sample size (303), limited demographic diversity, single-clinic origin.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Strip */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-rose-500 via-orange-500 to-amber-500 rounded-2xl animate-gradient" />
        <div className="relative grid grid-cols-2 sm:grid-cols-4 gap-4 p-6 sm:p-8 text-white">
          {[
            { label: 'CV ROC-AUC', value: '0.9109' },
            { label: 'Test Accuracy', value: '83.61%' },
            { label: 'Recall', value: '87.88%' },
            { label: 'Brier Score', value: '0.12' },
          ].map((stat, i) => (
            <div key={i} className="text-center glass rounded-xl p-4">
              <div className="text-xs sm:text-sm opacity-90 mb-1">{stat.label}</div>
              <div className="text-xl sm:text-2xl font-extrabold">{stat.value}</div>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="text-center py-6 sm:py-8">
        <div className="relative inline-block">
          <img src="/heart-illustration.svg" alt="" className="w-16 h-16 mx-auto mb-4 float-3d" />
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-gray-900 mb-4">Ready to Try It?</h2>
        <p className="text-gray-600 mb-6 max-w-lg mx-auto px-2">
          Enter clinical features and see the model's risk estimation with an explanation of the
          key contributing factors.
        </p>
        <button onClick={() => onNavigate('predict')} className="btn-primary group">
          <Activity className="w-5 h-5 inline mr-2 group-hover:animate-bounce" />
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
  gradient,
  bgGradient,
}: {
  icon: React.ReactNode
  title: string
  description: string
  gradient: string
  bgGradient: string
}) {
  return (
    <div className={`card-3d group cursor-default bg-gradient-to-br ${bgGradient}`}>
      <div className={`w-12 h-12 bg-gradient-to-br ${gradient} rounded-xl flex items-center justify-center mb-4 shadow-lg group-hover:scale-110 transition-transform duration-300 text-white`}>
        {icon}
      </div>
      <h3 className="font-bold text-gray-900 mb-2 group-hover:text-rose-600 transition-colors">{title}</h3>
      <p className="text-sm text-gray-600 leading-relaxed">{description}</p>
    </div>
  )
}
