import axios, { AxiosError } from 'axios';
import type { CaptionRecord, UploadResponse } from '@/types';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:5001/api';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
});

api.interceptors.response.use(
  (res) => res,
  (err: AxiosError) => {
    const message = (err.response?.data as { error?: string })?.error || err.message;
    return Promise.reject(new Error(message));
  }
);

export async function uploadImage(file: File, userId?: string): Promise<UploadResponse> {
  const form = new FormData();
  form.append('image', file);
  if (userId) form.append('userId', userId);

  const { data } = await api.post<UploadResponse>('/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

export async function fetchHistory(limit = 20, userId?: string): Promise<CaptionRecord[]> {
  const params = new URLSearchParams();
  params.set('limit', String(limit));
  if (userId) params.set('userId', userId);

  const { data } = await api.get<{ success: boolean; data: CaptionRecord[] }>(`/history?${params}`);
  return data.data;
}

export async function deleteHistoryRecord(id: string): Promise<void> {
  await api.delete(`/history/${id}`);
}

export async function fetchCaptions(page = 1, limit = 20) {
  const { data } = await api.get(`/captions?page=${page}&limit=${limit}`);
  return data;
}

export default api;
