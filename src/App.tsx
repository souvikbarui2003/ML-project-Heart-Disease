import { useState } from 'react'
import LandingPage from './pages/LandingPage'
import PredictPage from './pages/PredictPage'
import DashboardPage from './pages/DashboardPage'
import ResearchPage from './pages/ResearchPage'
import PwaInstallBanner from './components/PwaInstallBanner'
import { Heart, Activity, BarChart3, BookOpen, Menu, X } from 'lucide-react'

export type Page = 'home' | 'predict' | 'dashboard' | 'research'

export default function App() {
  const [page, setPage] = useState<Page>('home')
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const navItems: { page: Page; icon: React.ReactNode; label: string }[] = [
    { page: 'home', icon: <Heart className="w-5 h-5" />, label: 'Home' },
    { page: 'predict', icon: <Activity className="w-5 h-5" />, label: 'Predict' },
    { page: 'dashboard', icon: <BarChart3 className="w-5 h-5" />, label: 'Dashboard' },
    { page: 'research', icon: <BookOpen className="w-5 h-5" />, label: 'Research' },
  ]

  const handleNav = (p: Page) => {
    setPage(p)
    setMobileMenuOpen(false)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-blue-50">
      {/* Desktop Navigation */}
      <nav className="sticky top-0 z-50 bg-white/90 backdrop-blur-lg border-b border-gray-100 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-14 sm:h-16">
            <button
              onClick={() => handleNav('home')}
              className="flex items-center gap-2 text-primary-700 hover:text-primary-800 transition-colors"
            >
              <Heart className="w-5 h-5 sm:w-6 sm:h-6 fill-red-500 text-red-500" />
              <span className="font-bold text-base sm:text-lg">HeartGuard ML</span>
            </button>

            {/* Desktop Nav */}
            <div className="hidden sm:flex items-center gap-1">
              {navItems.map(item => (
                <button
                  key={item.page}
                  onClick={() => handleNav(item.page)}
                  className={`flex items-center gap-1.5 px-3 lg:px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    page === item.page
                      ? 'bg-primary-100 text-primary-700'
                      : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  {item.icon}
                  {item.label}
                </button>
              ))}
            </div>

            {/* Mobile Hamburger */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="sm:hidden p-2 rounded-lg text-gray-500 hover:text-gray-700 hover:bg-gray-50 transition-colors"
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Mobile Dropdown Menu */}
        {mobileMenuOpen && (
          <div className="sm:hidden border-t border-gray-100 bg-white/95 backdrop-blur-lg">
            <div className="px-4 py-2 space-y-1">
              {navItems.map(item => (
                <button
                  key={item.page}
                  onClick={() => handleNav(item.page)}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${
                    page === item.page
                      ? 'bg-primary-100 text-primary-700'
                      : 'text-gray-600 hover:bg-gray-50'
                  }`}
                >
                  {item.icon}
                  {item.label}
                </button>
              ))}
            </div>
          </div>
        )}
      </nav>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
        {page === 'home' && <LandingPage onNavigate={handleNav} />}
        {page === 'predict' && <PredictPage />}
        {page === 'dashboard' && <DashboardPage />}
        {page === 'research' && <ResearchPage onNavigate={handleNav} />}
      </main>

      {/* PWA Install Banner */}
      <PwaInstallBanner />

      {/* Mobile Bottom Tab Bar */}
      <nav className="sm:hidden fixed bottom-0 left-0 right-0 z-50 bg-white/95 backdrop-blur-lg border-t border-gray-200 safe-area-bottom">
        <div className="flex items-center justify-around py-2">
          {navItems.map(item => (
            <button
              key={item.page}
              onClick={() => handleNav(item.page)}
              className={`flex flex-col items-center gap-1 px-3 py-1.5 rounded-lg transition-all min-w-[60px] ${
                page === item.page
                  ? 'text-primary-600'
                  : 'text-gray-400'
              }`}
            >
              {item.icon}
              <span className="text-[10px] font-medium">{item.label}</span>
            </button>
          ))}
        </div>
      </nav>

      {/* Footer Disclaimer - hidden on mobile (bottom tab bar present) */}
      <footer className="hidden sm:block border-t border-gray-100 bg-white/50 mt-16 pb-20 sm:pb-0">
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

      {/* Mobile Footer Spacer */}
      <div className="sm:hidden h-20" />
    </div>
  )
}
