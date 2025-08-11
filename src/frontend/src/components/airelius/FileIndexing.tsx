import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2, Upload, Database, CheckCircle } from "lucide-react";

const FileIndexing = () => {
  const [isIndexing, setIsIndexing] = useState(false);
  const [indexedFiles, setIndexedFiles] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);

  const indexBackendFiles = async () => {
    setIsIndexing(true);
    setError(null);
    
    try {
      const response = await fetch("/api/v1/airelius/pfu/index/files", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          patterns: ["/src/backend/**/*.py"],
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setIndexedFiles(data.indexed_files || 0);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to index files");
    } finally {
      setIsIndexing(false);
    }
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Index Backend Files
          </CardTitle>
          <CardDescription>
            Upload and index backend files for AI context and planning
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Index backend files to provide AI with context about available components and functionality.
          </p>
          
          <Button 
            onClick={indexBackendFiles} 
            disabled={isIndexing}
            className="w-full"
          >
            {isIndexing ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Indexing Files...
              </>
            ) : (
              <>
                <Upload className="mr-2 h-4 w-4" />
                Index Backend Files
              </>
            )}
          </Button>
          
          {indexedFiles > 0 && (
            <div className="flex items-center gap-2 text-sm text-green-600 bg-green-50 p-3 rounded-md">
              <CheckCircle className="h-4 w-4" />
              Successfully indexed {indexedFiles} files
            </div>
          )}
          
          {error && (
            <div className="text-sm text-red-600 bg-red-50 p-3 rounded-md">
              {error}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default FileIndexing;
