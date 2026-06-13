import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { useAuth } from "../hooks/useAuth";


function RegisterPage() {
  const navigate = useNavigate();
  const { register } = useAuth();
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    confirm_password: "",
  });
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
      await register(formData);
      navigate("/login");
    } catch (err) {
      const data = err.response?.data;
      setError(data?.email?.[0] || data?.confirm_password?.[0] || data?.detail || "Registration failed.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section className="auth-panel">
      <h1>Register</h1>
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
          Email
          <input
            name="email"
            onChange={handleChange}
            required
            type="email"
            value={formData.email}
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
        <label>
          Confirm Password
          <input
            name="confirm_password"
            onChange={handleChange}
            required
            type="password"
            value={formData.confirm_password}
          />
        </label>
        {error && <p className="form-error">{error}</p>}
        <button disabled={submitting} type="submit">
          {submitting ? "Registering..." : "Register"}
        </button>
      </form>
      <p>
        Already have an account? <Link to="/login">Login</Link>
      </p>
    </section>
  );
}

export default RegisterPage;
