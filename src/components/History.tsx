import { useEffect, useState } from 'react';
import { HistoryIcon, Trash2, Loader2 } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { fetchHistory, deleteHistoryRecord } from '@/services/api';
import type { CaptionRecord } from '@/types';

interface HistoryProps {
  refreshTrigger: number;
}

export default function History({ refreshTrigger }: HistoryProps) {
  const [records, setRecords] = useState<CaptionRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const data = await fetchHistory(50);
      setRecords(data);
    } catch (err) {
      console.error('Failed to load history', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshTrigger]);

  const handleDelete = async (id: string) => {
    setDeletingId(id);
    try {
      await deleteHistoryRecord(id);
      setRecords((prev) => prev.filter((r) => r._id !== id));
    } catch (err) {
      console.error('Delete failed', err);
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <Card className="w-full">
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <HistoryIcon className="w-5 h-5" />
            Recent History
          </h2>
          <Button variant="ghost" size="sm" onClick={load} disabled={loading}>
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Refresh'}
          </Button>
        </div>

        {records.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-8">
            No captions generated yet. Upload an image to get started.
          </p>
        ) : (
          <div className="grid grid-cols-1 gap-3">
            {records.map((r) => {
              const clean = r.predictedCaption.replace('startseq', '').replace('endseq', '').trim();
              return (
                <div
                  key={r._id}
                  className="flex items-start justify-between rounded-lg border p-4 hover:bg-muted/50 transition"
                >
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-sm truncate">"{clean}"</p>
                    <div className="mt-2 flex items-center gap-2 text-xs text-muted-foreground">
                      <Badge variant="outline" className="text-[10px]">
                        {r.imageUrl}
                      </Badge>
                      <span>{new Date(r.createdAt).toLocaleString()}</span>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="ml-2 shrink-0"
                    onClick={() => handleDelete(r._id)}
                    disabled={deletingId === r._id}
                  >
                    {deletingId === r._id ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Trash2 className="w-4 h-4 text-destructive" />
                    )}
                  </Button>
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
