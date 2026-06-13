import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { useAuth } from "../hooks/useAuth";


function LoginPage() {
  const navigate = useNavigate();
  const { login } = useAuth();
  const [formData, setFormData] = useState({ username: "", password: "" });
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  function handleChange(event) {
    setFormData({
      ...formData,
      [event.target.name]: event.target.value,
    });
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setSubmitting(true);

    try {
      await login(formData);
      navigate("/dashboard");
    } catch {
      setError("Unable to log in with those credentials.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section className="auth-panel">
      <h1>Login</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Username
          <input
            name="username"
            onChange={handleChange}
            required
            type="text"
            value={formData.username}
          />
        </label>
        <label>
          Password
          <input
            name="password"
            onChange={handleChange}
            required
            type="password"
            value={formData.password}
          />
        </label>
        {error && <p className="form-error">{error}</p>}
        <button disabled={submitting} type="submit">
          {submitting ? "Logging in..." : "Login"}
        </button>
      </form>
      <p>
        Need an account? <Link to="/register">Register</Link>
      </p>
    </section>
  );
}

export default LoginPage;
