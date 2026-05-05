import { useState, useCallback } from 'react';
import Upload from '@/components/Upload';
import CaptionDisplay from '@/components/CaptionDisplay';
import History from '@/components/History';
import type { UploadResponse } from '@/types';

export default function Home() {
  const [result, setResult] = useState<UploadResponse['data'] | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const handleUploadSuccess = useCallback((data: UploadResponse['data']) => {
    setResult(data);
    setRefreshKey((k) => k + 1);
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="max-w-5xl mx-auto px-4 py-5 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">ImageCaption AI</h1>
            <p className="text-sm text-muted-foreground">
              VGG16 + LSTM neural network captioning, now production-ready
            </p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-5xl mx-auto px-4 py-8 space-y-6">
        <Upload onUploadSuccess={handleUploadSuccess} />

        {result && (
          <CaptionDisplay
            caption={result.caption}
            confidence={result.confidence}
            imageName={result.imageName}
            createdAt={result.createdAt}
          />
        )}

        <History refreshTrigger={refreshKey} />
      </main>

      {/* Footer */}
      <footer className="border-t mt-12 py-6 text-center text-sm text-muted-foreground">
        <p>Image Captioning with VGG16 + LSTM on Flickr8k · MERN Stack Demo</p>
      </footer>
    </div>
  );
}
