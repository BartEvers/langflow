import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent } from "@/components/ui/card";
import { Loader2, Send, X, Brain, Play, CheckCircle, XCircle, Upload, Database, RefreshCw, MessageSquare } from "lucide-react";
import useFlowStore from "@/stores/flowStore";


interface Message {
  id: string;
  type: 'user' | 'ai';
  content: string;
  timestamp: Date;
  plan?: any;
  isExecuting?: boolean;
  executionResult?: any;
  llm_response?: string;
}

const AireliusChat = () => {
  const [isHovered, setIsHovered] = useState(false);
  const hoverTimeoutRef = useRef<NodeJS.Timeout>();
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasError, setHasError] = useState(false);
  
  const currentFlow = useFlowStore((state) => state.currentFlow);
  const flowId = useFlowStore((state) => state.currentFlow?.id);
  
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Safe render function to prevent object rendering crashes
  const safeRender = (value: any): string => {
    if (value === null || value === undefined) return '';
    if (typeof value === 'string') return value;
    if (typeof value === 'number' || typeof value === 'boolean') return String(value);
    if (typeof value === 'object') {
      try {
        return JSON.stringify(value, null, 2);
      } catch {
        return '[Complex Object]';
      }
    }
    return String(value);
  };

  const extractInnerMonologue = (llmResponse: string): string => {
    if (!llmResponse || typeof llmResponse !== 'string') return '';
    
    const startTag = '<inner_monologue>';
    const endTag = '</inner_monologue>';
    
    const startIndex = llmResponse.indexOf(startTag);
    const endIndex = llmResponse.indexOf(endTag);
    
    if (startIndex === -1 || endIndex === -1 || startIndex >= endIndex) {
      return '';
    }
    
    // Extract content between tags, excluding the tags themselves
    const content = llmResponse.substring(startIndex + startTag.length, endIndex).trim();
    return content;
  };

  // Global error handler
  const handleGlobalError = (error: Error, errorInfo: any) => {
    console.error("Global error in AireliusChat:", error, errorInfo);
    setError(`Unexpected error: ${error.message}`);
    setHasError(true);
  };

  // Set up global error handling
  useEffect(() => {
    const handleError = (event: ErrorEvent) => {
      handleGlobalError(event.error || new Error(event.message), { 
        source: event.filename, 
        lineno: event.lineno, 
        colno: event.colno 
      });
    };

    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      if (event.reason) handleGlobalError(event.reason, { type: 'unhandledRejection' });
    };

    window.addEventListener('error', handleError);
    window.addEventListener('unhandledrejection', handleUnhandledRejection);

    return () => {
      window.removeEventListener('error', handleError);
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
    };
  }, []);

  const addMessage = (type: 'user' | 'ai', content: string, plan?: any, llm_response?: string) => {
    // Safety check: ensure content is a string
    const safeContent = safeRender(content);
    
    const newMessage: Message = {
      id: Date.now().toString(),
      type,
      content: safeContent,
      timestamp: new Date(),
      plan,
      llm_response,
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const generatePlan = async (prompt: string) => {
    if (!prompt.trim() || !flowId) return;

    setIsGenerating(true);
    setError(null);

    try {
      const response = await fetch("/api/v1/airelius/pfu/plan", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: prompt,
          flow_id: flowId,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      let data;
      try {
        data = await response.json();
        console.log("Received plan response:", data); // Debug logging
      } catch (jsonError) {
        console.error("JSON parse error:", jsonError); // Debug logging
        throw new Error("Invalid JSON response from server");
      }

      // Validate response structure
      if (!data || typeof data !== 'object') {
        console.error("Invalid response structure:", data); // Debug logging
        throw new Error("Invalid response format from server");
      }

      const plan = data.plan;
      console.log("Extracted plan:", plan); // Debug logging
      console.log("Plan structure details:", {
        hasPlan: !!plan,
        planType: typeof plan,
        planKeys: plan ? Object.keys(plan) : [],
        operationsType: plan?.operations ? typeof plan.operations : 'none',
        operationsLength: plan?.operations ? (Array.isArray(plan.operations) ? plan.operations.length : 'not array') : 'none'
      });
      
      // Debug the full response structure
      console.log("Full response data:", data);
      console.log("Response keys:", Object.keys(data));
      console.log("LLM response field:", data.llm_response);
      
      // Show internal monologue if available
      if (data.llm_response) {
        console.log("=== AI INTERNAL MONOLOGUE (from data) ===");
        console.log(data.llm_response);
        console.log("=== END MONOLOGUE ===");
        
        // Also show the extracted inner monologue content
        const extractedMonologue = extractInnerMonologue(data.llm_response);
        if (extractedMonologue) {
          console.log("=== EXTRACTED INNER MONOLOGUE CONTENT ===");
          console.log(extractedMonologue);
          console.log("=== END EXTRACTED CONTENT ===");
        }
      }
      
      if (plan?.llm_response) {
        console.log("=== AI INTERNAL MONOLOGUE (from plan) ===");
        console.log(plan.llm_response);
        console.log("=== END MONOLOGUE ===");
      }
      
      // Debug operations structure
      if (plan?.operations && Array.isArray(plan.operations)) {
        console.log("Operations array:", plan.operations);
        plan.operations.forEach((op: any, index: number) => {
          console.log(`Operation ${index}:`, {
            type: typeof op,
            keys: op ? Object.keys(op) : [],
            operation: op?.operation,
            description: op?.description,
            validation: op?.validation
          });
        });
      }
      
      // Validate plan structure
      if (!plan || typeof plan !== 'object') {
        console.warn("Received invalid plan structure:", plan);
        // Even if plan parsing failed, show the raw LLM response
        addMessage('ai', `I've created a plan for: "${prompt}", but the format was unexpected. However, here's my reasoning:`, null, data.llm_response);
        return;
      }
      
      // Add AI response with plan and llm_response
      addMessage('ai', `I've created a plan for: "${prompt}"`, plan, data.llm_response);
      
    } catch (err) {
      console.error("Error generating plan:", err);
      const errorMessage = err instanceof Error ? err.message : "Failed to generate plan";
      setError(errorMessage);
      addMessage('ai', `Sorry, I encountered an error: ${errorMessage}`);
    } finally {
      setIsGenerating(false);
    }
  };

  const executePlan = async (plan: any) => {
    if (!plan || !flowId) return;

    setIsExecuting(true);
    setError(null);

    try {
      // Console log the API call being made in copy-pasteable format
      console.log("=== EXECUTING PFU PLAN - API CALL DETAILS ===");
      console.log("URL: /api/v1/airelius/pfu/execute");
      console.log("Method: POST");
      console.log("Flow ID:", flowId);
      console.log("Max Steps: 10");
      console.log("Request Body:");
      console.log(JSON.stringify({
        plan: plan,
        flow_id: flowId,
        max_steps: 10,
      }, null, 2));
      console.log("=== END API CALL DETAILS ===");
      
      const response = await fetch("/api/v1/airelius/pfu/execute", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          plan: plan,
          flow_id: flowId,
          max_steps: 10,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      let data;
      try {
        data = await response.json();
        console.log("Received execution response:", data); // Debug logging
      } catch (jsonError) {
        console.error("JSON parse error in execution:", jsonError); // Debug logging
        throw new Error("Invalid JSON response from server");
      }

      // Validate response structure
      if (!data || typeof data !== 'object') {
        console.error("Invalid execution response structure:", data); // Debug logging
        throw new Error("Invalid response format from server");
      }
      
      // Update the message with execution result
      setMessages(prev => prev.map(msg => 
        msg.plan && plan && JSON.stringify(msg.plan) === JSON.stringify(plan)
          ? { ...msg, executionResult: data, isExecuting: false }
          : msg
      ));
      
      // Add success message
      addMessage('ai', `Plan executed successfully! Check the results above.`);
      
    } catch (err) {
      console.error("Error executing plan:", err);
      const errorMessage = err instanceof Error ? err.message : "Failed to execute plan";
      setError(errorMessage);
      
      // Update the message with error result
      setMessages(prev => prev.map(msg => 
        msg.plan && plan && JSON.stringify(msg.plan) === JSON.stringify(plan)
          ? { ...msg, isExecuting: false, executionResult: { error: errorMessage } }
          : msg
      ));
    } finally {
      setIsExecuting(false);
    }
  };

  const executeSingleOperation = async (operation: any, description: string) => {
    if (!operation || !flowId) return;

    try {
      // Console log the API call being made in copy-pasteable format
      console.log("=== EXECUTING SINGLE OPERATION - API CALL DETAILS ===");
      console.log("URL:", `/api/v1/flows/${flowId}/ops`);
      console.log("Method: POST");
      console.log("Flow ID:", flowId);
      console.log("Description:", description);
      console.log("Request Body:");
      console.log(JSON.stringify({
        operations: [operation],
      }, null, 2));
      console.log("=== END API CALL DETAILS ===");
      
      const response = await fetch(`/api/v1/flows/${flowId}/ops`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          operations: [operation],
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      addMessage('ai', `✅ Operation executed: ${description}`);
      
      // Refresh the flow to show changes
      // This would typically trigger a flow refresh in the main app
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to execute operation";
      addMessage('ai', `❌ Operation failed: ${description} - ${errorMessage}`);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isGenerating) return;

    const prompt = inputValue.trim();
    setInputValue("");
    
    // Add user message
    addMessage('user', prompt);
    
    // Generate plan
    await generatePlan(prompt);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };



  const indexBackendFiles = async () => {
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
      addMessage('ai', `Successfully indexed ${data.indexed_files || 0} backend files for AI context.`);
    } catch (err) {
      addMessage('ai', `Failed to index files: ${err instanceof Error ? err.message : "Unknown error"}`);
    }
  };

  // Safety check - if there's a critical error, show error state
  if (hasError) {
    return (
      <div className="fixed bottom-6 left-1/2 transform -translate-x-1/2 z-50">
        <Card className="shadow-lg border-2 border-red-200 bg-red-50">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-red-600">
              <XCircle className="h-4 w-4" />
              <span className="text-sm font-medium">Airelius Chat Error</span>
            </div>
            <p className="text-xs text-red-500 mt-2">
              The chat component encountered an error. Please refresh the page to restart.
            </p>
            <Button
              size="sm"
              variant="outline"
              onClick={() => window.location.reload()}
              className="mt-3 h-7 px-3 text-xs"
            >
              Refresh Page
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <>
      {/* Hover-Expanded Chat Conversation */}
      {isHovered && (
        <div 
          className="fixed bottom-20 left-1/2 transform -translate-x-1/2 z-40"
          onMouseEnter={() => {
            if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current);
            setIsHovered(true);
          }}
          onMouseLeave={() => {
            hoverTimeoutRef.current = setTimeout(() => setIsHovered(false), 300);
          }}
        >
          <Card className="w-full max-w-6xl max-h-[70vh] shadow-lg border-2 border-primary/20 bg-background/95 backdrop-blur-sm">
            <div className="p-4">
              <div className="flex justify-between items-center mb-3 pb-2 border-b">
                <span className="text-sm font-medium">AI Flow Assistant</span>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={indexBackendFiles}
                  className="h-6 px-2 text-xs"
                >
                  <Database className="h-3 w-3 mr-1" />
                  Index Files
                </Button>
              </div>
              <div className="overflow-y-auto max-h-[50vh]">
                <div className="space-y-3">
                {!Array.isArray(messages) || messages.length === 0 ? (
                  <div className="text-center text-muted-foreground py-8">
                    <Brain className="h-12 w-12 mx-auto mb-3 opacity-50" />
                    <p>Start a conversation to generate AI-powered flow modifications</p>
                    <p className="text-sm mt-1">Try asking: "Add error handling to my flow"</p>
                  </div>
                ) : (
                  messages.filter(msg => msg && typeof msg === 'object' && msg.id).map((message) => (
                  <div key={message.id} className={`flex gap-3 ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`flex gap-3 ${message.type === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                      <div className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs ${
                        message.type === 'user' ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'
                      }`}>
                        {message.type === 'user' ? 'U' : 'AI'}
                      </div>
                      <div className={`max-w-[80%] p-3 rounded-lg ${
                        message.type === 'user' 
                          ? 'bg-primary text-primary-foreground' 
                          : 'bg-muted text-foreground'
                      }`}>
                        <div className="text-sm">{safeRender(message.content)}</div>
                        {message.plan && (
                          <div className="mt-3 p-3 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg">
                            <div className="flex items-center gap-2 mb-2">
                              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                              <span className="text-xs font-semibold text-blue-700">AI Generated Plan</span>
                            </div>
                            
                            {message.plan && message.plan.parsing_error && (
                              <div className="mb-3 p-2 bg-yellow-50 border border-yellow-200 rounded">
                                <div className="text-xs font-medium text-yellow-800 mb-1">⚠️ Plan Parsing Warning:</div>
                                <div className="text-xs text-yellow-700 mb-2">
                                  {safeRender(message.plan.parsing_error)}
                                </div>
                                <div className="text-xs text-yellow-600">
                                  <strong>What this means:</strong> The AI generated a plan but couldn't format it properly. 
                                  You can still see the AI's reasoning below, and you may be able to manually execute the suggested changes.
                                </div>
                              </div>
                            )}
                            
                            {message.plan && message.plan.objective && (
                              <div className="mb-3">
                                <div className="text-xs font-medium text-blue-800 mb-1">Objective:</div>
                                <div className="text-xs text-blue-700 bg-white/60 p-2 rounded border-l-2 border-blue-300">
                                  {safeRender(message.plan.objective)}
                                </div>
                              </div>
                            )}
                            
                            {message.plan && message.plan.current_state_analysis && (
                              <div className="mb-3">
                                <div className="text-xs font-medium text-blue-800 mb-1">Current State Analysis:</div>
                                <div className="text-xs text-blue-700 bg-white/60 p-2 rounded border-l-2 border-blue-300">
                                  {safeRender(message.plan.current_state_analysis)}
                                </div>
                              </div>
                            )}
                            
                            {message.plan && message.plan.required_changes && (
                              <div className="mb-3">
                                <div className="text-xs font-medium text-blue-800 mb-1">Required Changes:</div>
                                <div className="text-xs text-blue-700 bg-white/60 p-2 rounded border-l-2 border-blue-300">
                                  {safeRender(message.plan.required_changes)}
                                </div>
                              </div>
                            )}
                            
                            {message.plan && message.plan.comprehensive_plan && (
                              <div className="mb-3">
                                <div className="text-xs font-medium text-blue-800 mb-1">Comprehensive Plan:</div>
                                <div className="text-xs text-blue-700 bg-white/60 p-2 rounded border-l-2 border-blue-300 max-h-40 overflow-y-auto">
                                  <div className="whitespace-pre-wrap">
                                    {safeRender(message.plan.comprehensive_plan)}
                                  </div>
                                </div>
                              </div>
                            )}
                            
                            {message.plan && message.plan.execution_strategy && (
                              <div className="mb-3">
                                <div className="text-xs font-medium text-blue-800 mb-1">Execution Strategy:</div>
                                <div className="text-xs text-blue-700 bg-white/60 p-2 rounded border-l-2 border-blue-300">
                                  {safeRender(message.plan.execution_strategy)}
                                </div>
                              </div>
                            )}
                            
                            {message.plan && message.plan.validation_strategy && (
                              <div className="mb-3">
                                <div className="text-xs font-medium text-blue-800 mb-1">Validation Strategy:</div>
                                <div className="text-xs text-blue-700 bg-white/60 p-2 rounded border-l-2 border-blue-300">
                                  {safeRender(message.plan.validation_strategy)}
                                </div>
                              </div>
                            )}
                            
                            {/* Show AI reasoning when plan parsing fails but we have LLM response */}
                            {message.plan && message.plan.parsing_error && message.plan.llm_response && (
                              <div className="mb-3">
                                <div className="text-xs font-medium text-blue-800 mb-1">AI Reasoning:</div>
                                <div className="text-xs text-blue-700 bg-white/60 p-2 rounded border-l-2 border-blue-300 max-h-32 overflow-y-auto">
                                  <div className="whitespace-pre-wrap">
                                    {safeRender(message.plan.llm_response)}
                                  </div>
                                </div>
                              </div>
                            )}
                            
                            {/* Show inner monologue if available */}
                            {message.plan && message.plan.inner_monologue && (
                              <div className="mb-3">
                                <div className="text-xs font-medium text-blue-800 mb-1">AI Strategy:</div>
                                <div className="text-xs text-blue-700 bg-white/60 p-2 rounded border-l-2 border-blue-300 max-h-32 overflow-y-auto">
                                  <div className="whitespace-pre-wrap">
                                    {safeRender(message.plan.inner_monologue)}
                                  </div>
                                </div>
                              </div>
                            )}
                            
                            {message.plan && message.plan.llm_response && (
                              <div className="mb-3">
                                <div className="text-xs font-medium text-blue-800 mb-1">AI Internal Monologue:</div>
                                <div className="text-xs text-blue-600 bg-white/60 p-2 rounded border-l-2 border-blue-300 font-mono">
                                  {safeRender(message.plan.llm_response)}
                                </div>
                              </div>
                            )}
                            
                            {message.llm_response && (
                              <div className="mb-3">
                                <div className="text-xs font-medium text-blue-800 mb-1 flex items-center gap-2">
                                  <Brain className="h-3 w-3" />
                                  AI Internal Monologue
                                </div>
                                <div className="text-xs text-blue-600 bg-white/60 p-2 rounded border-l-2 border-blue-300 font-mono max-h-32 overflow-y-auto">
                                  {extractInnerMonologue(message.llm_response)}
                                </div>
                              </div>
                            )}
                            
                            {message.plan && message.plan.operations && message.plan.operations.length > 0 && (
                              <div className="mb-4">
                                <div className="text-xs font-medium text-blue-800 mb-2">Operations:</div>
                                <div className="space-y-3">
                                  {message.plan.operations.map((op, index) => (
                                    <div key={index} className="bg-white/60 p-3 rounded border border-blue-200">
                                      <div className="flex items-center justify-between mb-2">
                                        <div className="text-xs font-medium text-blue-800 mb-2">
                                          Step {index + 1}: {op.operation?.op || 'Unknown Operation'}
                                        </div>
                                        <button
                                          className="px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
                                          onClick={() => executeSingleOperation(op.operation, op.description || `Step ${index + 1}`)}
                                        >
                                          Execute
                                        </button>
                                      </div>
                                      
                                      {op.description && (
                                        <div className="text-xs text-blue-700 mb-2">
                                          <span className="font-medium">Description:</span> {safeRender(op.description)}
                                        </div>
                                      )}
                                      
                                      {op.reasoning && (
                                        <div className="text-xs text-blue-700 mb-2">
                                          <span className="font-medium">Reasoning:</span> {safeRender(op.reasoning)}
                                        </div>
                                      )}
                                      
                                      {op.component_analysis && (
                                        <div className="text-xs text-blue-700 mb-2">
                                          <span className="font-medium">Component Analysis:</span> {safeRender(op.component_analysis)}
                                        </div>
                                      )}
                                      
                                      {op.validation && (
                                        <div className="text-xs text-blue-700 mb-2">
                                          <span className="font-medium">Validation:</span> {safeRender(op.validation)}
                                        </div>
                                      )}
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                            
                            {message.plan && message.plan.operations && Array.isArray(message.plan.operations) && message.plan.operations.length > 0 && (
                              <div className="flex justify-end">
                                <Button
                                  size="sm"
                                  onClick={() => executePlan(message.plan)}
                                  disabled={isExecuting}
                                  className="h-7 px-3 text-xs bg-blue-600 hover:bg-blue-700 text-white border-0"
                                >
                                  {isExecuting ? (
                                    <Loader2 className="h-3 w-3 animate-spin mr-1" />
                                  ) : (
                                    <Play className="h-3 w-3 mr-1" />
                                  )}
                                  Execute All Steps
                                </Button>
                              </div>
                            )}
                            
                            {/* Show helpful message when no operations are available */}
                            {message.plan && (!message.plan.operations || !Array.isArray(message.plan.operations) || message.plan.operations.length === 0) && (
                              <div className="mt-3 p-2 bg-blue-50 border border-blue-200 rounded">
                                <div className="text-xs text-blue-700 mb-2">
                                  <strong>No executable operations available.</strong> 
                                  {message.plan.parsing_error ? 
                                    " The AI's plan couldn't be parsed into executable steps. " :
                                    " The AI didn't generate any executable operations. "
                                  }
                                  You can try rephrasing your request or ask the AI to be more specific about what changes to make.
                                </div>
                                <div className="flex gap-2">
                                  <Button
                                    size="sm"
                                    onClick={() => {
                                      const lastUserMessage = messages.slice().reverse().find(msg => msg.type === 'user');
                                      if (lastUserMessage) {
                                        generatePlan(lastUserMessage.content);
                                      }
                                    }}
                                    className="h-6 px-2 text-xs bg-blue-600 hover:bg-blue-700 text-white border-0"
                                  >
                                    <RefreshCw className="h-3 w-3 mr-1" />
                                    Retry
                                  </Button>
                                  <Button
                                    size="sm"
                                    onClick={() => {
                                      const lastUserMessage = messages.slice().reverse().find(msg => msg.type === 'user');
                                      if (lastUserMessage) {
                                        const retryPrompt = lastUserMessage.content + " (Please provide a clear, structured plan with executable operations in valid JSON format)";
                                        generatePlan(retryPrompt);
                                      }
                                    }}
                                    className="h-6 px-2 text-xs bg-green-600 hover:bg-green-700 text-white border-0"
                                  >
                                    <MessageSquare className="h-3 w-3 mr-1" />
                                    Retry with Format Hint
                                  </Button>
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                        
                        {message.executionResult && (
                          <div className="mt-3 p-3 bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-lg">
                            <div className="flex items-center gap-2 mb-2">
                              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                              <span className="text-xs font-semibold text-green-700">Execution Result</span>
                            </div>
                            
                            {message.executionResult.error ? (
                              <div className="flex items-center gap-2 text-red-600 bg-red-50 p-2 rounded border-l-2 border-red-300">
                                <XCircle className="h-4 w-4" />
                                <span className="text-xs">Error: {safeRender(message.executionResult.error)}</span>
                              </div>
                            ) : (
                              <div className="space-y-2">
                                {message.executionResult && message.executionResult.execution_summary && (
                                  <div className="bg-white/60 p-2 rounded border-l-2 border-green-300">
                                    <div className="text-xs font-medium text-green-800 mb-1">Summary:</div>
                                    <div className="grid grid-cols-3 gap-2 text-xs">
                                      <div className="text-center">
                                        <div className="text-green-600 font-semibold">
                                          {safeRender(message.executionResult.execution_summary.total_steps) || '0'}
                                        </div>
                                        <div className="text-green-500">Total Steps</div>
                                      </div>
                                      <div className="text-center">
                                        <div className="text-green-600 font-semibold">
                                          {safeRender(message.executionResult.execution_summary.successful_steps) || '0'}
                                        </div>
                                        <div className="text-green-500">Successful</div>
                                      </div>
                                      <div className="text-center">
                                        <div className="text-green-600 font-semibold">
                                          {safeRender(message.executionResult.execution_summary.failed_steps) || '0'}
                                        </div>
                                        <div className="text-green-500">Failed</div>
                                      </div>
                                    </div>
                                  </div>
                                )}
                                
                                {message.executionResult && message.executionResult.step_results && Array.isArray(message.executionResult.step_results) && message.executionResult.step_results.length > 0 && (
                                  <div className="bg-white/60 p-2 rounded border-l-2 border-green-300">
                                    <div className="text-xs font-medium text-green-800 mb-2">Step Details:</div>
                                    <div className="space-y-1">
                                      {message.executionResult.step_results.map((step: any, index: number) => (
                                        <div key={index} className="flex items-center gap-2 text-xs">
                                          {step?.success ? (
                                            <CheckCircle className="h-3 w-3 text-green-500" />
                                          ) : (
                                            <XCircle className="h-3 w-3 text-red-500" />
                                          )}
                                          <span className={step?.success ? 'text-green-700' : 'text-red-700'}>
                                            Step {index + 1}: {safeRender(step?.operation) || 'Unknown operation'}
                                          </span>
                                          {step?.message && (
                                            <span className="text-green-600 ml-2">({safeRender(step.message)})</span>
                                          )}
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))
                )}
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Floating Chat Input Bar */}
      <div 
        className="fixed bottom-6 left-1/2 transform -translate-x-1/2 z-50"
        onMouseEnter={() => {
          if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current);
          setIsHovered(true);
        }}
        onMouseLeave={() => {
          hoverTimeoutRef.current = setTimeout(() => setIsHovered(false), 300);
        }}
      >
        <Card className="shadow-lg border-2 border-primary/20 bg-background/95 backdrop-blur-sm">
          <CardContent className="p-0">
            <form onSubmit={handleSubmit} className="flex items-center gap-2 p-2">
              <div className="flex items-center gap-2 text-sm text-muted-foreground px-2">
                <Brain className="h-4 w-4" />
                <span className="hidden sm:inline">
                  {!flowId ? "No Flow Loaded" : "AI Flow Assistant"}
                </span>
              </div>
              
              <Textarea
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={!flowId ? "No flow loaded..." : "Ask me to modify your flow..."}
                className="min-w-[500px] max-w-[800px] resize-none border-0 focus-visible:ring-0 focus-visible:ring-offset-0 bg-transparent"
                rows={1}
                disabled={isGenerating || !flowId}
              />
              
              <div className="flex items-center gap-1">
                <Button
                  type="submit"
                  size="sm"
                  disabled={isGenerating || !inputValue.trim() || !flowId}
                  className="h-8 w-8 p-0"
                >
                  {isGenerating ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                </Button>
                

              </div>
            </form>
            
            {/* Error Display */}
            {error && (
              <div className="px-3 pb-2">
                <div className="flex items-center gap-2 text-red-600 text-xs bg-red-50 p-2 rounded border border-red-200">
                  <XCircle className="h-3 w-3" />
                  <span className="truncate">{safeRender(error)}</span>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => setError(null)}
                    className="h-5 w-5 p-0 ml-auto hover:bg-red-100"
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>


    </>
  );
};

export default AireliusChat;
