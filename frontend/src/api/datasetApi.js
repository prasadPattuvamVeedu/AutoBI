import axiosClient from "./axiosClient";


export const datasetApi = {
  list: () => axiosClient.get("/datasets/"),
  upload: (formData) => axiosClient.post("/datasets/upload/", formData),
  detail: (id) => axiosClient.get(`/datasets/${id}/`),
  preview: (id) => axiosClient.get(`/datasets/${id}/preview/`),
  profile: (id) => axiosClient.get(`/datasets/${id}/profile/`),
  remove: (id) => axiosClient.delete(`/datasets/${id}/`),
};
