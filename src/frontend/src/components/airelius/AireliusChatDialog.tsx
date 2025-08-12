import { useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  Loader2, 
  Play, 
  CheckCircle, 
  XCircle, 
  Upload, 
  Database, 
  X,
  Brain,
  User,
  Clock
} from "lucide-react";

interface Message {
  id: string;
  type: 'user' | 'ai';
  content: string;
  timestamp: Date;
  plan?: any;
  isExecuting?: boolean;
  executionResult?: any;
}

interface AireliusChatDialogProps {
  messages: Message[];
  onExecutePlan: (plan: any) => void;
  onIndexFiles: () => void;
  isExecuting: boolean;
  error: string | null;
  onClose: () => void;
}

const AireliusChatDialog = ({
  messages,
  onExecutePlan,
  onIndexFiles,
  isExecuting,
  error,
  onClose
}: AireliusChatDialogProps) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const renderPlan = (plan: any) => {
    if (!plan) return null;

    return (
      <Card className="mt-3 border-l-4 border-l-primary">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium">Generated Plan</CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="space-y-2 text-sm">
            <div>
              <span className="font-medium">Objective:</span>
              <p className="text-muted-foreground mt-1">{plan.objective}</p>
            </div>
            
            <div>
              <span className="font-medium">Strategy:</span>
              <p className="text-muted-foreground mt-1">{plan.execution_strategy}</p>
            </div>
            
            <div>
              <span className="font-medium">Operations ({plan.operations?.length || 0}):</span>
              <div className="space-y-1 mt-1">
                {plan.operations?.map((op: any, index: number) => (
                  <div key={index} className="text-xs bg-muted p-2 rounded">
                    <strong>Step {op.step}:</strong> {op.description}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  };

  const renderExecutionResult = (result: any) => {
    if (!result) return null;

    return (
      <Card className="mt-3 border-l-4 border-l-green-500">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-green-700 flex items-center gap-2">
            <CheckCircle className="h-4 w-4" />
            Execution Completed
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="space-y-2 text-sm">
            <div className="grid grid-cols-3 gap-4 text-xs">
              <div className="text-center">
                <div className="font-medium text-green-600">{result.execution_summary?.total_steps || 0}</div>
                <div className="text-muted-foreground">Total Steps</div>
              </div>
              <div className="text-center">
                <div className="font-medium text-green-600">{result.execution_summary?.successful_steps || 0}</div>
                <div className="text-muted-foreground">Successful</div>
              </div>
              <div className="text-center">
                <div className="font-medium text-red-600">{result.execution_summary?.failed_steps || 0}</div>
                <div className="text-muted-foreground">Failed</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  };

  const renderMessage = (message: Message) => {
    const isUser = message.type === 'user';
    
    return (
      <div
        key={message.id}
        className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}
      >
        <div className={`flex gap-3 max-w-[80%] ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
          {/* Avatar */}
          <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
            isUser ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'
          }`}>
            {isUser ? <User className="h-4 w-4" /> : <Brain className="h-4 w-4" />}
          </div>
          
          {/* Message Content */}
          <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
            <div className={`rounded-lg px-3 py-2 ${
              isUser 
                ? 'bg-primary text-primary-foreground' 
                : 'bg-muted text-foreground'
            }`}>
              <p className="text-sm">{message.content}</p>
            </div>
            
            {/* Timestamp */}
            <div className="flex items-center gap-1 mt-1 text-xs text-muted-foreground">
              <Clock className="h-3 w-3" />
              {formatTime(message.timestamp)}
            </div>
            
            {/* Plan Display */}
            {!isUser && message.plan && renderPlan(message.plan)}
            
            {/* Execution Result */}
            {!isUser && message.executionResult && renderExecutionResult(message.executionResult)}
            
            {/* Action Buttons */}
            {!isUser && message.plan && !message.executionResult && (
              <div className="mt-2 flex gap-2">
                <Button
                  size="sm"
                  onClick={() => onExecutePlan(message.plan)}
                  disabled={isExecuting}
                  className="h-7 text-xs"
                >
                  {isExecuting ? (
                    <>
                      <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                      Executing...
                    </>
                  ) : (
                    <>
                      <Play className="mr-1 h-3 w-3" />
                      Execute Plan
                    </>
                  )}
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center p-4 bg-black/50">
      <Card className="w-full max-w-4xl max-h-[80vh] flex flex-col">
        <CardHeader className="flex-shrink-0 border-b">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              AI Flow Assistant
            </CardTitle>
            <div className="flex items-center gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={onIndexFiles}
                className="h-8 text-xs"
              >
                <Database className="mr-1 h-3 w-3" />
                Index Files
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={onClose}
                className="h-8 w-8 p-0"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        
        <div className="flex-1 p-4 overflow-y-auto">
          <div className="space-y-4">
            {messages.length === 0 ? (
              <div className="text-center text-muted-foreground py-8">
                <Brain className="h-12 w-12 mx-auto mb-3 opacity-50" />
                <p>Start a conversation to generate AI-powered flow modifications</p>
                <p className="text-sm mt-1">Try asking: "Add error handling to my flow"</p>
              </div>
            ) : (
              messages.map(renderMessage)
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>
        
        {error && (
          <div className="p-4 border-t bg-red-50">
            <div className="flex items-center gap-2 text-red-600 text-sm">
              <XCircle className="h-4 w-4" />
              {error}
            </div>
          </div>
        )}
      </Card>
    </div>
  );
};

export default AireliusChatDialog;
