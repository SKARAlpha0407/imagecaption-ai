import { Sparkles } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface CaptionDisplayProps {
  caption: string;
  confidence: number;
  imageName: string;
  createdAt: string;
}

export default function CaptionDisplay({ caption, confidence, imageName, createdAt }: CaptionDisplayProps) {
  const cleanCaption = caption.replace('startseq', '').replace('endseq', '').trim();

  return (
    <Card className="w-full border-primary/20">
      <CardContent className="p-6">
        <div className="flex items-center gap-2 mb-3">
          <Sparkles className="w-5 h-5 text-primary" />
          <h3 className="text-lg font-semibold">Generated Caption</h3>
          <Badge variant="secondary" className="ml-auto">
            {Math.round(confidence * 100)}% confidence
          </Badge>
        </div>

        <p className="text-xl leading-relaxed font-medium text-foreground">
          "{cleanCaption}"
        </p>

        <div className="mt-4 flex items-center gap-3 text-xs text-muted-foreground">
          <span className="truncate max-w-[200px]">{imageName}</span>
          <span>·</span>
          <span>{new Date(createdAt).toLocaleString()}</span>
        </div>
      </CardContent>
    </Card>
  );
}
