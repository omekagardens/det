import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Theory from './pages/Theory';
import Results from './pages/Results';
import Falsifiers from './pages/Falsifiers';
import Novelty from './pages/Novelty';
import Simulator from './pages/Simulator';
import Footer from './components/Footer';
import './styles/App.css';
import './styles/pages.css';

function App() {
  const [darkMode, setDarkMode] = useState(true);

  useEffect(() => {
    document.body.className = darkMode ? 'dark-mode' : 'light-mode';
  }, [darkMode]);

  return (
    <Router>
      <div className={`app ${darkMode ? 'dark' : 'light'}`}>
        <Navbar darkMode={darkMode} setDarkMode={setDarkMode} />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/theory" element={<Theory />} />
            <Route path="/results" element={<Results />} />
            <Route path="/falsifiers" element={<Falsifiers />} />
            <Route path="/novelty" element={<Novelty />} />
            <Route path="/simulator" element={<Simulator />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
