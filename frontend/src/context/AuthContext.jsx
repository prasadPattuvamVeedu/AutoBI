import { createContext, useEffect, useMemo, useState } from "react";

import axiosClient from "../api/axiosClient";
import { getMe, loginUser, registerUser } from "../api/authApi";


export const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [accessToken, setAccessToken] = useState(() => localStorage.getItem("accessToken"));
  const [refreshToken, setRefreshToken] = useState(() => localStorage.getItem("refreshToken"));
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(Boolean(accessToken));

  useEffect(() => {
    if (accessToken) {
      axiosClient.defaults.headers.common.Authorization = `Bearer ${accessToken}`;
      localStorage.setItem("accessToken", accessToken);
    } else {
      delete axiosClient.defaults.headers.common.Authorization;
      localStorage.removeItem("accessToken");
    }
  }, [accessToken]);

  useEffect(() => {
    if (refreshToken) {
      localStorage.setItem("refreshToken", refreshToken);
    } else {
      localStorage.removeItem("refreshToken");
    }
  }, [refreshToken]);

  useEffect(() => {
    async function loadUser() {
      if (!accessToken) {
        setLoading(false);
        return;
      }

      try {
        const response = await getMe();
        setUser(response.data);
      } catch {
        logout();
      } finally {
        setLoading(false);
      }
    }

    loadUser();
  }, [accessToken]);

  async function login(credentials) {
    const response = await loginUser(credentials);
    setAccessToken(response.data.access);
    setRefreshToken(response.data.refresh);
    axiosClient.defaults.headers.common.Authorization = `Bearer ${response.data.access}`;

    const meResponse = await getMe();
    setUser(meResponse.data);
    return meResponse.data;
  }

  async function register(data) {
    const response = await registerUser(data);
    return response.data;
  }

  function logout() {
    setAccessToken(null);
    setRefreshToken(null);
    setUser(null);
  }

  const value = useMemo(
    () => ({
      user,
      isAuthenticated: Boolean(accessToken),
      loading,
      login,
      register,
      logout,
    }),
    [accessToken, loading, user]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
