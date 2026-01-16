import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';

function Navbar({ darkMode, setDarkMode }) {
  const location = useLocation();

  const links = [
    { path: '/', label: 'Home' },
    { path: '/theory', label: 'Theory' },
    { path: '/results', label: 'Results' },
    { path: '/falsifiers', label: 'Falsifiers' },
    { path: '/novelty', label: 'Novelty' },
    { path: '/simulator', label: '3D Simulator' },
  ];

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <Link to="/" className="navbar-logo">
          <span className="logo-det">DET</span>
          <span className="logo-version">v6.3</span>
        </Link>

        <div className="navbar-links">
          {links.map((link) => (
            <Link
              key={link.path}
              to={link.path}
              className={`navbar-link ${location.pathname === link.path ? 'active' : ''}`}
            >
              {link.label}
              {location.pathname === link.path && (
                <motion.div
                  className="navbar-indicator"
                  layoutId="navbar-indicator"
                  transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                />
              )}
            </Link>
          ))}
        </div>

        <button
          className="theme-toggle"
          onClick={() => setDarkMode(!darkMode)}
          aria-label="Toggle dark mode"
        >
          {darkMode ? (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="5"/>
              <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
            </svg>
          ) : (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
            </svg>
          )}
        </button>
      </div>
    </nav>
  );
}

export default Navbar;
