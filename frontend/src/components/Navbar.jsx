import { Link } from "react-router-dom";

import { useAuth } from "../hooks/useAuth";


function Navbar() {
  const { isAuthenticated, logout, user } = useAuth();

  return (
    <nav className="navbar">
      <Link className="navbar-title" to="/dashboard">AutoBI AI</Link>
      <div className="navbar-links">
        {isAuthenticated ? (
          <>
            <Link to="/dashboard">Dashboard</Link>
            <span>{user?.username}</span>
            <button type="button" onClick={logout}>Logout</button>
          </>
        ) : (
          <>
            <Link to="/login">Login</Link>
            <Link to="/register">Register</Link>
          </>
        )}
      </div>
    </nav>
  );
}

export default Navbar;
