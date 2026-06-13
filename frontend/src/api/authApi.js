import axiosClient from "./axiosClient";


export function registerUser(data) {
  return axiosClient.post("/auth/register/", data);
}

export function loginUser(credentials) {
  return axiosClient.post("/auth/login/", credentials);
}

export function getMe() {
  return axiosClient.get("/auth/me/");
}

export function refreshToken(refresh) {
  return axiosClient.post("/auth/token/refresh/", { refresh });
}
