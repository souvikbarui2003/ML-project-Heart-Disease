import { useState } from 'react'
import LandingPage from './pages/LandingPage'
import PredictPage from './pages/PredictPage'
import DashboardPage from './pages/DashboardPage'
import ResearchPage from './pages/ResearchPage'
import PwaInstallBanner from './components/PwaInstallBanner'
import { Heart, Activity, BarChart3, BookOpen } from 'lucide-react'

export type Page = 'home' | 'predict' | 'dashboard' | 'research'

export default function App() {
  const [page, setPage] = useState<Page>('home')

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-blue-50">
      {/* Navigation */}
      <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-lg border-b border-gray-100 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <button
              onClick={() => setPage('home')}
              className="flex items-center gap-2 text-primary-700 hover:text-primary-800 transition-colors"
            >
              <Heart className="w-6 h-6 fill-red-500 text-red-500" />
              <span className="font-bold text-lg">HeartGuard ML</span>
            </button>

            <div className="flex items-center gap-1">
              <NavLink active={page === 'home'} onClick={() => setPage('home')} icon={<Heart className="w-4 h-4" />} label="Home" />
              <NavLink active={page === 'predict'} onClick={() => setPage('predict')} icon={<Activity className="w-4 h-4" />} label="Predict" />
              <NavLink active={page === 'dashboard'} onClick={() => setPage('dashboard')} icon={<BarChart3 className="w-4 h-4" />} label="Dashboard" />
              <NavLink active={page === 'research'} onClick={() => setPage('research')} icon={<BookOpen className="w-4 h-4" />} label="Research" />
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {page === 'home' && <LandingPage onNavigate={setPage} />}
        {page === 'predict' && <PredictPage />}
        {page === 'dashboard' && <DashboardPage />}
        {page === 'research' && <ResearchPage onNavigate={setPage} />}
      </main>

      {/* PWA Install Banner */}
      <PwaInstallBanner />

      {/* Footer Disclaimer */}
      <footer className="border-t border-gray-100 bg-white/50 mt-16">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-xs text-gray-500 text-center leading-relaxed max-w-3xl mx-auto">
            <strong className="text-gray-600">Medical Disclaimer:</strong> This application is a
            machine-learning research and educational tool. It is <strong>not</strong> a medical
            diagnostic device. Predictions should not replace evaluation by a qualified healthcare
            professional. This model was trained on a limited dataset (303 patients) and may not
            generalize across all populations. Always consult a physician for medical advice.
          </p>
        </div>
      </footer>
    </div>
  )
}

function NavLink({ active, onClick, icon, label }: { active: boolean; onClick: () => void; icon: React.ReactNode; label: string }) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
        active ? 'bg-primary-100 text-primary-700' : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
      }`}
    >
      {icon}
      {label}
    </button>
  )
}
