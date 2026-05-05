export interface CaptionRecord {
  _id: string;
  imageUrl: string;
  predictedCaption: string;
  actualCaptions: string[];
  userId: string | null;
  confidence: number;
  modelVersion: string;
  createdAt: string;
}

export interface UploadResponse {
  success: boolean;
  data: {
    id: string;
    caption: string;
    confidence: number;
    imageName: string;
    createdAt: string;
  };
}

export interface ApiError {
  success: false;
  error: string;
}
