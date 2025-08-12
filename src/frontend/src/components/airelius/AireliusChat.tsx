import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent } from "@/components/ui/card";
import { Loader2, Send, X, Brain, Play, CheckCircle, XCircle, Upload, Database, RefreshCw, MessageSquare, MessageCircle, Zap } from "lucide-react";
import useFlowStore from "@/stores/flowStore";
import useFlowsManagerStore from "@/stores/flowsManagerStore";
import { useTypesStore } from "@/stores/typesStore";
import { useAddComponent } from "@/hooks/use-add-component";

/**
 * AireliusChat Component
 * 
 * This component handles AI-powered chat and agent planning for Langflow flows.
 * 
 * Expected Response Format for LLM:
 * The LLM should return responses in one of these structured formats:
 * 
 * 1. Simple Response:
 * {
 *   "complexity_level": "simple|moderate|complex",
 *   "reasoning": "AI's reasoning about the request",
 *   "required_steps": [
 *     {
 *       "name": "step_name",
 *       "description": "What this step does",
 *       "type": "direct_execution|planning|validation",
 *       "priority": "high|medium|low"
 *     }
 *   ]
 * }
 * 
 * 2. Agent Plan Response:
 * {
 *   "objective": "What we want to achieve",
 *   "current_state_analysis": "Analysis of current flow state",
 *   "required_changes": "What needs to change",
 *   "execution_strategy": "How to implement changes",
 *   "operations": [
 *     {
 *       "operation": { "op": "operation_type", ... },
 *       "description": "What this operation does",
 *       "reasoning": "Why this operation is needed"
 *     }
 *   ]
 * }
 * 
 * The component will automatically detect and format these responses beautifully.
 */


interface Message {
  id: string;
  type: 'user' | 'ai';
  content: string;
  timestamp: Date;
  plan?: any;
  isExecuting?: boolean;
  executionResult?: any;
  llm_response?: string;
  isStreaming?: boolean;
  streamedContent?: string;
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
  const [isStreaming, setIsStreaming] = useState(false);
  const [stepResults, setStepResults] = useState<{[key: number]: any}>({});
  const [currentStep, setCurrentStep] = useState(0);
  const [finalPlan, setFinalPlan] = useState<any>(null);
  const [chatMode, setChatMode] = useState<'plan' | 'streaming'>('plan');
  const [abortController, setAbortController] = useState<AbortController | null>(null);
  
  const currentFlow = useFlowsManagerStore((state) => state.currentFlow);
  const flowId = useFlowsManagerStore((state) => state.currentFlowId);
  
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  
  // Get access to types store and addComponent hook
  const templates = useTypesStore((state) => state.templates);
  const addComponent = useAddComponent();

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (chatContainerRef.current && messages.length > 0) {
      // Small delay to ensure the message is fully rendered
      const timeoutId = setTimeout(() => {
        if (chatContainerRef.current) {
          chatContainerRef.current.scrollTo({
            top: chatContainerRef.current.scrollHeight,
            behavior: 'smooth'
          });
        }
      }, 50);
      
      return () => clearTimeout(timeoutId);
    }
  }, [messages]);

  // Auto-scroll to bottom when streaming content updates
  useEffect(() => {
    if (chatContainerRef.current && isStreaming) {
      const scrollToBottom = () => {
        if (chatContainerRef.current) {
          chatContainerRef.current.scrollTo({
            top: chatContainerRef.current.scrollHeight,
            behavior: 'smooth'
          });
        }
      };
      
      // Scroll immediately
      scrollToBottom();
      
      // Also scroll after a small delay to ensure content is rendered
      const timeoutId = setTimeout(scrollToBottom, 100);
      
      return () => clearTimeout(timeoutId);
    }
  }, [isStreaming, messages.filter(msg => msg.isStreaming).length]);

  // Auto-scroll to bottom when chat dialog is opened
  useEffect(() => {
    if (isHovered && chatContainerRef.current && messages.length > 0) {
      const timeoutId = setTimeout(() => {
        scrollToBottom();
      }, 100);
      
      return () => clearTimeout(timeoutId);
    }
  }, [isHovered, messages.length]);

  // Auto-scroll to bottom when chat mode changes
  useEffect(() => {
    if (chatContainerRef.current && messages.length > 0) {
      const timeoutId = setTimeout(() => {
        scrollToBottom();
      }, 100);
      
      return () => clearTimeout(timeoutId);
    }
  }, [chatMode]);

  // Helper function to scroll to bottom
  const scrollToBottom = () => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTo({
        top: chatContainerRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  };

  // Function to add components when LLM outputs component tags
  const handleComponentTags = (llmOutput: string) => {
    if (!flowId || !currentFlow) {
      console.warn('No flow loaded, cannot add components');
      return;
    }
    
    if (Object.keys(templates).length === 0) {
      console.warn('No templates loaded, cannot add components');
      return;
    }
    
    console.log('🔍 LLM Output:', llmOutput);
    console.log('🔍 Available templates:', Object.keys(templates));
    
    const componentTags = extractComponentTags(llmOutput);
    console.log('🔍 Extracted component tags:', componentTags);
    
    componentTags.forEach(tag => {
      console.log(`🔍 Looking for component: "${tag}"`);
      console.log(`🔍 Template exists:`, !!templates[tag]);
      
      if (templates[tag]) {
        try {
          // Add the component to the flow
          addComponent(templates[tag], tag);
          console.log(`✅ Added component: ${tag}`);
        } catch (error) {
          console.error(`❌ Failed to add component ${tag}:`, error);
        }
      } else {
        console.warn(`⚠️ Component type "${tag}" not found in templates`);
        // Try to find similar component names
        const similarComponents = Object.keys(templates).filter(name => 
          name.toLowerCase().includes(tag.toLowerCase()) || 
          tag.toLowerCase().includes(name.toLowerCase())
        );
        if (similarComponents.length > 0) {
          console.log(`💡 Similar components found:`, similarComponents);
        }
      }
    });
  };

  // Click outside handler to close dialog immediately
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (isHovered) {
        const target = event.target as Element;
        
        // Check if click is outside both the chat dialog and input
        const isOutsideDialog = !target.closest('[data-chat-dialog]');
        const isOutsideInput = !target.closest('[data-chat-input]');
        
        if (isOutsideDialog && isOutsideInput) {
          console.log('Click outside detected, closing dialog');
          setIsHovered(false);
          if (hoverTimeoutRef.current) {
            clearTimeout(hoverTimeoutRef.current);
          }
        }
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isHovered]);

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

  const formatLLMResponse = (data: any): string => {
    if (!data || typeof data !== 'object') return 'Invalid response data';
    
    let formatted = '';
    
    // Add Inner Monologue
    if (data.inner_monologue) {
      formatted += `**Inner Monologue:** ${data.inner_monologue}\n\n`;
    }
    
    // Add Planning Approach
    if (data.planning_approach) {
      formatted += `**Planning Approach:** ${data.planning_approach}\n\n`;
    }
    
    // Add Key Considerations
    if (data.key_considerations && Array.isArray(data.key_considerations)) {
      formatted += `**Key Considerations:**\n`;
      data.key_considerations.forEach((consideration: string, index: number) => {
        formatted += `${index + 1}. ${consideration}\n`;
      });
      formatted += '\n';
    }
    
    // Add Risk Assessment
    if (data.risk_assessment) {
      formatted += `**Risk Assessment:** ${data.risk_assessment}\n\n`;
    }
    
    // Add Response if it exists
    if (data.response) {
      formatted += `**Response:** ${data.response}\n\n`;
    }
    
    // If no structured data found, return the raw data
    if (formatted === '') {
      return JSON.stringify(data, null, 2);
    }
    
    return formatted.trim();
  };

  const formatLLMOutputForDisplay = (text: string): JSX.Element => {
    if (!text || typeof text !== 'string') return <span>No output available</span>;
    
    // Split by lines and format each line
    const lines = text.split('\n');
    const formattedLines = lines.map((line, index) => {
      if (line.startsWith('**') && line.endsWith('**')) {
        // Bold headers
        return (
          <div key={index} className="font-bold text-blue-700 mb-2">
            {line.replace(/\*\*/g, '')}
          </div>
        );
      } else if (line.match(/^\d+\./)) {
        // Numbered lists
        return (
          <div key={index} className="ml-4 mb-1">
            {line}
          </div>
        );
      } else if (line.trim() === '') {
        // Empty lines
        return <div key={index} className="mb-2"></div>;
      } else {
        // Regular text
        return (
          <div key={index} className="mb-1">
            {line}
          </div>
        );
      }
    });
    
    return <div>{formattedLines}</div>;
  };

  // Enhanced response formatter that handles various response structures
  const formatResponseForDisplay = (response: any): JSX.Element => {
    if (!response) return <span>No response available</span>;
    
    // If it's already a string, try to parse it as JSON first
    if (typeof response === 'string') {
      try {
        const parsed = JSON.parse(response);
        return formatResponseForDisplay(parsed);
      } catch {
        // If it's not JSON, treat as plain text
        return <div className="whitespace-pre-wrap">{response}</div>;
      }
    }
    
    // If it's not an object, convert to string
    if (typeof response !== 'object') {
      return <span>{String(response)}</span>;
    }
    
    // Handle nested response structures (common in some APIs)
    if (response.response && typeof response.response === 'object') {
      return formatResponseForDisplay(response.response);
    }
    
    if (response.data && typeof response.data === 'object') {
      return formatResponseForDisplay(response.data);
    }
    
    // Handle different response structures
    const elements: JSX.Element[] = [];
    
    // Handle complexity_level
    if (response.complexity_level) {
      elements.push(
        <div key="complexity" className="p-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-sm font-semibold text-gray-700">🎯 Complexity Assessment</span>
            <span className="text-xs text-gray-600 bg-gray-100 px-2 py-1 rounded border">
              {response.complexity_level === 'simple' ? '🟢' : 
               response.complexity_level === 'moderate' ? '🟡' : '🔴'} 
              {response.complexity_level.charAt(0).toUpperCase() + response.complexity_level.slice(1)}
            </span>
          </div>
        </div>
      );
    }
    
    // Handle reasoning
    if (response.reasoning) {
      elements.push(
        <div key="reasoning" className="p-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-sm font-semibold text-gray-700">🧠 AI Reasoning</span>
          </div>
          <div className="text-sm text-gray-700 leading-relaxed">
            {response.reasoning}
          </div>
        </div>
      );
    }
    
    // Handle required_steps
    if (response.required_steps && Array.isArray(response.required_steps)) {
      elements.push(
        <div key="steps" className="p-3">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-sm font-semibold text-gray-700">📋 Required Steps</span>
            <span className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded">
              {response.required_steps.length} step{response.required_steps.length !== 1 ? 's' : ''}
            </span>
          </div>
          <div className="space-y-3">
            {response.required_steps.map((step: any, index: number) => (
              <div key={index}>
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {step.type && (
                      <div className="mt-2">
                        <span className="text-xs text-gray-600 bg-gray-100 px-2 py-1 rounded border block">
                        {index + 1}. {step.type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </span>
                      </div>
                    )}
                    {step.priority && (
                      <span className="px-2 py-1 mt-2 text-xs rounded font-medium bg-gray-100 text-gray-700 border border-gray-200">
                        Prio: {step.priority.charAt(0).toUpperCase() + step.priority.slice(1)} {step.priority === 'high' ? '🔴' : 
                         step.priority === 'medium' ? '🟡' : '🔵'} 
                      </span>
                    )}
                  </div>
                </div>
                
                {step.description && (
                  <div className="text-sm text-gray-700 px-1">
                    {step.description}
                  </div>
                )}
                

              </div>
            ))}
          </div>
        </div>
      );
    }
    
    // Handle operations (if present)
    if (response.operations && Array.isArray(response.operations)) {
      elements.push(
        <div key="operations" className="p-3">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-sm font-semibold text-gray-700">⚡ Component Operations</span>
            <span className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded">
              {response.operations.length} operation{response.operations.length !== 1 ? 's' : ''}
            </span>
          </div>
          <div className="space-y-3">
            {response.operations.map((op: any, index: number) => (
              <div key={index}>
                <div className="flex items-center justify-between mb-2">
                  <div className="text-sm font-medium text-gray-800">
                    Operation {index + 1}: {op.operation?.op || 'Unknown Operation'}
                  </div>
                  <span className="px-3 py-1 text-xs bg-blue-100 text-blue-700 rounded border">
                    Component Tag
                  </span>
                </div>
                
                {op.description && (
                  <div className="text-sm text-gray-700 mb-2">
                    <span className="font-medium">Description:</span> {op.description}
                  </div>
                )}
                
                {op.reasoning && (
                  <div className="text-sm text-gray-700 mb-2">
                    <span className="font-medium">Reasoning:</span> {op.reasoning}
                  </div>
                )}
                
                {/* Display component tags if present */}
                {op.response && extractComponentTags(op.response).length > 0 && (
                  <div className="mt-2">
                    <div className="text-xs font-medium text-gray-700 mb-1">Component Tags:</div>
                    <div className="space-y-1">
                      {extractComponentTags(op.response).map((tag, tagIndex) => (
                        <div key={tagIndex} className="text-xs bg-blue-50 text-blue-700 px-2 py-1 rounded border">
                          {tag}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      );
    }
    
    // Handle other common fields
    const otherFields = ['objective', 'current_state_analysis', 'required_changes', 'execution_strategy', 'validation_strategy'];
    otherFields.forEach(field => {
      if (response[field]) {
        elements.push(
          <div key={field} className="p-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-sm font-semibold text-gray-700">
                {field.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </span>
            </div>
            <div className="text-sm text-gray-700 leading-relaxed">
              {response[field]}
            </div>
          </div>
        );
      }
    });
    
    // Handle any remaining fields that might be useful
    const remainingFields = Object.keys(response).filter(key => 
      !['complexity_level', 'reasoning', 'required_steps', 'operations', ...otherFields].includes(key) &&
      response[key] && 
      typeof response[key] !== 'object' &&
      response[key] !== ''
    );
    
    if (remainingFields.length > 0) {
      elements.push(
        <div key="other" className="p-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-sm font-semibold text-gray-700">Additional Information</span>
          </div>
          <div className="space-y-2">
            {remainingFields.map(field => (
              <div key={field} className="text-sm">
                <span className="font-medium text-gray-700 capitalize">
                  {field.replace(/_/g, ' ')}:
                </span>
                <span className="ml-2 text-gray-600">
                  {String(response[field])}
                </span>
              </div>
            ))}
          </div>
        </div>
      );
    }
    
    // If no structured content was found, show a fallback
    if (elements.length === 0) {
      return (
        <div className="p-3">
          <div className="text-sm text-gray-600">
            <div className="font-medium mb-2">Raw Response:</div>
            <pre className="text-xs bg-gray-100 p-2 rounded border overflow-x-auto">
              {JSON.stringify(response, null, 2)}
            </pre>
          </div>
        </div>
      );
    }
    
    return <div>{elements}</div>;
  };

  // Helper function to extract component tags from text
  const extractComponentTags = (text: string): string[] => {
    if (!text || typeof text !== 'string') {
      return [];
    }
    
    // Look for simple component tags like <ComponentType> or <ComponentType>content</ComponentType>
    // This regex will match <ComponentType> and extract ComponentType
    const componentRegex = /<(\w+)>/g;
    
    const matches: string[] = [];
    let match;
    
    while ((match = componentRegex.exec(text)) !== null) {
      // match[1] contains the captured group (the component type)
      matches.push(match[1]);
    }
    
    // Remove duplicates and filter out common HTML tags
    const filteredMatches = Array.from(new Set(matches)).filter(tag => 
      !['html', 'head', 'body', 'div', 'span', 'p', 'br', 'hr', 'img', 'a', 'ul', 'ol', 'li', 'table', 'tr', 'td', 'th'].includes(tag.toLowerCase())
    );
    
    return filteredMatches;
  };

  // Helper function to create a message with structured content
  const createStructuredMessage = (content: any, prefix: string = ''): string => {
    if (typeof content === 'string') {
      try {
        const parsed = JSON.parse(content);
        // If it's valid JSON, return a simple prefix since formatting will happen in display
        return prefix ? `${prefix} (Structured Response)` : 'Structured Response';
      } catch {
        // If it's not JSON, return as is
        return prefix ? `${prefix}: ${content}` : content;
      }
    }
    
    // If it's already an object, return prefix
    if (typeof content === 'object' && content !== null) {
      return prefix ? `${prefix} (Structured Response)` : 'Structured Response';
    }
    
    // Fallback
    return prefix ? `${prefix}: ${String(content)}` : String(content);
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
      // Cleanup abort controller
      if (abortController) {
        abortController.abort();
      }
    };
  }, [abortController]);

  // Cleanup function to remove duplicate messages
  const cleanupDuplicateMessages = () => {
    setMessages(prev => {
      const seen = new Set();
      const uniqueMessages = prev.filter(msg => {
        if (!msg || !msg.id || seen.has(msg.id)) {
          return false;
        }
        seen.add(msg.id);
        return true;
      });
      
      // If we removed any duplicates, log it
      if (uniqueMessages.length !== prev.length) {
        console.log(`🧹 Cleaned up ${prev.length - uniqueMessages.length} duplicate messages`);
      }
      
      return uniqueMessages;
    });
  };

  // Clean up duplicates when messages change
  useEffect(() => {
    if (messages.length > 0) {
      cleanupDuplicateMessages();
    }
  }, [messages.length]);

  // Also clean up duplicates when messages array changes (more comprehensive)
  useEffect(() => {
    const interval = setInterval(() => {
      if (messages.length > 0) {
        cleanupDuplicateMessages();
      }
    }, 1000); // Check every second

    return () => clearInterval(interval);
  }, []);

  const [messageCounter, setMessageCounter] = useState(0);
  
  const addMessage = (type: 'user' | 'ai', content: string, plan?: any, llm_response?: string, isStreaming?: boolean) => {
    // Safety check: ensure content is a string
    const safeContent = safeRender(content);
    
    // Use timestamp + counter + random string for guaranteed unique IDs
    const timestamp = Date.now();
    const randomStr = Math.random().toString(36).substr(2, 9);
    const newId = `msg-${timestamp}-${messageCounter}-${randomStr}`;
    setMessageCounter(prev => prev + 1);
    
    const newMessage: Message = {
      id: newId,
      type,
      content: safeContent,
      timestamp: new Date(),
      plan,
      llm_response,
      isStreaming,
      streamedContent: isStreaming ? '' : safeContent,
    };
    
    // Check for duplicates before adding
    setMessages(prev => {
      const hasDuplicate = prev.some(msg => msg.id === newId);
      if (hasDuplicate) {
        console.warn(`⚠️ Duplicate message ID detected: ${newId}, regenerating...`);
        // Regenerate ID and retry
        const timestamp = Date.now();
        const randomStr = Math.random().toString(36).substr(2, 9);
        const regeneratedId = `msg-${timestamp}-${messageCounter}-${randomStr}`;
        newMessage.id = regeneratedId;
        return [...prev, newMessage];
      }
      return [...prev, newMessage];
    });
    
    // Scroll to bottom after adding new message
    setTimeout(() => {
      scrollToBottom();
    }, 50);
    
    return newId;
  };

  const updateStreamingMessage = (messageId: string, newContent: string) => {
    setMessages(prev => prev.map(msg => 
      msg.id === messageId 
        ? { 
            ...msg, 
            streamedContent: newContent,
            content: newContent, // Keep both in sync during streaming
            isStreaming: true // Ensure streaming state is maintained
          }
        : msg
    ));
    
    // Scroll to bottom to show streaming content
    setTimeout(() => {
      if (chatContainerRef.current) {
        chatContainerRef.current.scrollTo({
          top: chatContainerRef.current.scrollHeight,
          behavior: 'smooth'
        });
      }
    }, 50);
  };

  const stopStreaming = () => {
    if (abortController) {
      abortController.abort();
      setAbortController(null);
      setIsGenerating(false);
      console.log('🛑 Streaming stopped by user');
    }
  };

  const streamingChat = async (prompt: string) => {
    if (!prompt.trim() || !flowId) return;

    console.log(`🚀 Starting streaming chat with prompt: "${prompt}"`);
    setIsGenerating(true);
    setError(null);
    let aiMessageId: string | undefined;

    // Create abort controller for this streaming session
    const controller = new AbortController();
    setAbortController(controller);

    try {
      // Add user message
      const userId = addMessage('user', prompt);
      
      // Add initial AI message that will be streamed into
      aiMessageId = addMessage('ai', 'Thinking...', undefined, undefined, true);
      
      // Show initial loading state
      if (aiMessageId) {
        updateStreamingMessage(aiMessageId, 'Thinking...');
      }

      const response = await fetch("/api/v1/airelius/pfu/chat/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: prompt,
          flow_id: flowId,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      console.log(`📡 API response received, status: ${response.status}`);
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }

      console.log(`📖 Starting to read streaming response...`);

      let streamedContent = '';
      
      try {
        while (true) {
          // Check if streaming was aborted
          if (controller.signal.aborted) {
            console.log('🛑 Streaming aborted by user');
            if (aiMessageId) {
              updateStreamingMessage(aiMessageId, 'Streaming stopped by user.');
            }
            break;
          }

          const { done, value } = await reader.read();
          if (done) {
            console.log(`📖 Reader finished, done: ${done}`);
            // If we have content but didn't get a done signal, finalize the message
            if (streamedContent && aiMessageId) {
              console.log(`⚠️ Reader finished unexpectedly, finalizing message with content: "${streamedContent}"`);
              setMessages(prev => prev.map(msg => 
                msg.id === aiMessageId 
                  ? { ...msg, isStreaming: false, content: streamedContent, streamedContent: streamedContent }
                  : msg
              ));
            }
            break;
          }
          
          const text = new TextDecoder().decode(value);
          const lines = text.split('\n');
          //console.log(`📦 Received ${lines.length} lines, total bytes: ${value?.length || 0}`);
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                //console.log(`📦 Parsed streaming data:`, data);
                
                if (data.error) {
                  console.error('Streaming error:', data.error);
                  setError(data.error);
                  // Update the streaming message to show the error
                  if (aiMessageId) {
                    updateStreamingMessage(aiMessageId, `❌ Error: ${data.error}`);
                  }
                  break;
                }
                
                if (data.content) {
                  streamedContent += data.content;
                  console.log(`🔄 Streaming content: "${data.content}" -> Total: "${streamedContent}"`);
                  if (aiMessageId) {
                    updateStreamingMessage(aiMessageId, streamedContent);
                    // Small delay to make streaming visible (adjust this value for desired speed)
                    await new Promise(resolve => setTimeout(resolve, 5));
                  }
                } else if (data.content === '') {
                  // Handle empty content gracefully
                  console.log(`📭 Received empty content chunk`);
                }
                
                if (data.done) {
                  console.log(`✅ Streaming completed. Final content: "${streamedContent}"`);
                  // Mark message as no longer streaming and finalize content
                  if (aiMessageId) {
                    setMessages(prev => prev.map(msg => 
                      msg.id === aiMessageId 
                        ? { ...msg, isStreaming: false, content: streamedContent, streamedContent: streamedContent }
                        : msg
                    ));
                    
                    // Check for component tags and add components automatically
                    console.log('🔍 Final streamed content:', streamedContent);
                    console.log('🔍 Calling handleComponentTags with:', streamedContent);
                    handleComponentTags(streamedContent);
                    
                    // Final scroll to bottom
                    setTimeout(() => {
                      if (chatContainerRef.current) {
                        chatContainerRef.current.scrollTo({
                          top: chatContainerRef.current.scrollHeight,
                          behavior: 'smooth'
                        });
                      }
                    }, 100);
                  }
                  break;
                }
                
              } catch (e) {
                console.error('Failed to parse streaming data:', e, 'Raw line:', line);
                // Continue processing other lines even if one fails
              }
            } else if (line.trim()) {
              console.log(`📝 Non-data line received: "${line}"`);
            }
          }
        }
      } catch (streamError: any) {
        console.error('Error during streaming loop:', streamError);
        if (aiMessageId) {
          if (streamError.name === 'AbortError') {
            updateStreamingMessage(aiMessageId, 'Streaming stopped by user.');
          } else if (streamError.name === 'TypeError' && streamError.message.includes('network')) {
            updateStreamingMessage(aiMessageId, '❌ Network error. Please check your connection and try again.');
          } else {
            updateStreamingMessage(aiMessageId, `❌ Streaming error: ${streamError.message}`);
          }
        }
      } finally {
        reader.releaseLock();
        console.log(`🔒 Reader released, streaming loop completed`);
        
        // Safety check: ensure the message is properly finalized
        if (aiMessageId && streamedContent) {
          setMessages(prev => prev.map(msg => 
            msg.id === aiMessageId 
              ? { ...msg, isStreaming: false, content: streamedContent, streamedContent: streamedContent }
              : msg
          ));
          
          // Check for component tags in the final content
          handleComponentTags(streamedContent);
        }
      }
      
    } catch (error: any) {
      console.error('Streaming chat failed:', error);
      setError(error.message);
      // Update the streaming message to show the error
      if (aiMessageId) {
        updateStreamingMessage(aiMessageId, `❌ Failed to start streaming: ${error.message}`);
      }
    } finally {
      console.log(`🏁 Streaming chat finished, cleaning up...`);
      setIsGenerating(false);
    }
  };

  const generateAgentPlan = async (prompt: string) => {
    if (!prompt.trim() || !flowId) return;

    console.log(`🚀 Starting AI agent planning with prompt: "${prompt}"`);
    setIsGenerating(true);
    setIsStreaming(true);
    setError(null);
    setStepResults({});
    setCurrentStep(0);
    setFinalPlan(null);

    // Create abort controller for this streaming session
    const controller = new AbortController();
    setAbortController(controller);

    let aiMessageId: string | undefined;

    try {
      // Add initial message
      aiMessageId = addMessage('ai', `🚀 Starting to create a 7-step agent plan for: "${prompt}"...`, undefined, undefined, true);

      const response = await fetch("/api/v1/airelius/pfu/plan/step-by-step/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: prompt,
          flow_id: flowId,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      console.log(`📡 Plan API response received, status: ${response.status}`);
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }

      console.log(`📖 Starting to read plan streaming response...`);

      try {
        while (true) {
          // Check if streaming was aborted
          if (controller.signal.aborted) {
            console.log('🛑 Plan generation aborted by user');
                    if (aiMessageId) {
          updateStreamingMessage(aiMessageId, 'Agent plan generation stopped by user.');
        }
            break;
          }

          const { done, value } = await reader.read();
          if (done) {
            console.log(`📖 Plan reader finished, done: ${done}`);
            break;
          }
          
          const text = new TextDecoder().decode(value);
          const lines = text.split('\n');
          //console.log(`📦 Received ${lines.length} lines, total bytes: ${value?.length || 0}`);
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                //console.log(`📦 Parsed plan streaming data:`, data);
                
                if (data.error) {
                  console.error('Plan streaming error:', data.error);
                  setError(data.error);
                  if (aiMessageId) {
                    updateStreamingMessage(aiMessageId, `❌ Error: ${data.error}`);
                  }
                  break;
                }
                
                if (data.status === 'starting') {
                  console.log('🔄 Step starting:', data);
                  setCurrentStep(data.step);
                  setStepResults(prev => ({
                    ...prev,
                    [data.step]: { status: 'starting', message: data.message }
                  }));
                  
                  // Add a new message for each step starting
                  addMessage('ai', `Step ${data.step}: ${data.message}...`, undefined, undefined, false);
                  
                  console.log(`Step ${data.step} starting: ${data.message}`);
                }
                
                // Handle streaming content for each step
                if (data.type === 'streaming' && data.content) {
                  //console.log(`🔄 Streaming content for step ${data.step}:`, data.content);
                  
                  setStepResults(prev => {
                    const currentStep = prev[data.step] || { status: 'streaming', streamed_content: '' };
                    const newStreamedContent = (currentStep.streamed_content || '') + data.content;
                    
                    return {
                      ...prev,
                      [data.step]: {
                        ...currentStep,
                        streamed_content: newStreamedContent
                      }
                    };
                  });
                  
                  // Create or update streaming message for this step
                  setMessages(prev => {
                    const existingMessage = prev.find(msg => 
                      msg.content.includes(`Step ${data.step}:`) && msg.isStreaming
                    );
                    
                    if (existingMessage) {
                      // Update existing streaming message
                      return prev.map(msg => 
                        msg.id === existingMessage.id 
                          ? { ...msg, streamedContent: (msg.streamedContent || '') + data.content }
                          : msg
                      );
                    } else {
                      // Create new streaming message for this step with unique ID
                      const timestamp = Date.now();
                      const stepId = `step-${data.step}-${timestamp}-${Math.random().toString(36).substr(2, 9)}`;
                      const newMessage = {
                        id: stepId,
                        type: 'ai' as const,
                        content: `Step ${data.step}: Streaming LLM output...`,
                        timestamp: new Date(),
                        isStreaming: true,
                        streamedContent: data.content
                      };
                      return [...prev, newMessage];
                    }
                  });
                }
                
                // Handle step completion with streaming LLM output
                if (data.status === 'completed') {
                  console.log('✅ Step completed:', data);
                  console.log('🔍 Raw data.result:', data.result);
                  console.log('🔍 data.result type:', typeof data.result);
                  console.log('🔍 data.result keys:', data.result && typeof data.result === 'object' ? Object.keys(data.result) : 'Not an object');
                  
                  // Format the LLM output to be readable
                  let llmOutput = '';
                  if (data.result && typeof data.result === 'object') {
                    // Extract key information from the result
                    const result = data.result;
                    
                    if (result.response) {
                      // Parse the response field which contains the structured output
                      try {
                        const responseData = JSON.parse(result.response);
                        llmOutput = formatLLMResponse(responseData);
                      } catch (e) {
                        // If parsing fails, show the raw response
                        llmOutput = result.response;
                      }
                    } else if (result.inner_monologue) {
                      // Direct access to inner_monologue
                      llmOutput = formatLLMResponse(result);
                    } else {
                      // Fallback to showing the entire result
                      llmOutput = JSON.stringify(data.result, null, 2);
                    }
                  } else if (data.result && typeof data.result === 'string') {
                    // Try to parse string as JSON
                    try {
                      const parsed = JSON.parse(data.result);
                      llmOutput = formatLLMResponse(parsed);
                    } catch (e) {
                      llmOutput = data.result;
                    }
                  } else {
                    llmOutput = 'No output available';
                  }
                  console.log('🔍 Final llmOutput:', llmOutput);
                  
                  setStepResults(prev => {
                    const newStepResults = {
                      ...prev,
                      [data.step]: { 
                        status: 'completed', 
                        result: data.result,
                        message: data.message,
                        llm_output: llmOutput
                      }
                    };
                    console.log('🔍 Updated stepResults:', newStepResults);
                    return newStepResults;
                  });
                  
                  // Update the last message to show completion
                  setMessages(prev => {
                    const newMessages = [...prev];
                    if (newMessages.length > 0) {
                      const lastMessage = newMessages[newMessages.length - 1];
                      if (lastMessage.content.includes(`Step ${data.step}:`)) {
                        lastMessage.content = `Step ${data.step}: ${data.message} ✓`;
                      }
                    }
                    return newMessages;
                  });
                  
                  // Finalize any streaming messages for this step
                  setMessages(prev => prev.map(msg => 
                    msg.content.includes(`Step ${data.step}:`) && msg.isStreaming
                      ? { 
                          ...msg, 
                          isStreaming: false, 
                          content: `Step ${data.step}: ${data.message} ✓`,
                          streamedContent: `Step ${data.step}: ${data.message} ✓`,
                          llm_response: llmOutput
                        }
                      : msg
                  ));
                  
                                      // Check for component tags and add components automatically
                    console.log('🔍 LLM output for step:', llmOutput);
                    handleComponentTags(llmOutput);
                  
                  console.log(`Step ${data.step} completed: ${data.message}`, data.result);
                  console.log(`LLM Output extracted:`, llmOutput);
                }
                
                if (data.status === 'completed' && data.final_plan) {
                  console.log('🎯 Final plan received:', data);
                  setFinalPlan(data.final_plan);
                  setIsStreaming(false);
                  
                  // Add the AI's direct response
                  addMessage('ai', data.message, data, undefined, false);
                  
                                      // Check for component tags in the AI's response
                    if (data.message) {
                      console.log('🔍 AI response:', data.message);
                      handleComponentTags(data.message);
                    }
                  
                  console.log('AI response received:', data.message);
                  break;
                }
                
              } catch (e) {
                console.error('Failed to parse plan streaming data:', e, 'Raw line:', line);
                // Continue processing other lines even if one fails
              }
            } else if (line.trim()) {
              console.log(`📝 Non-data line received: "${line}"`);
            }
          }
        }
      } catch (streamError: any) {
        console.error('Error during plan streaming loop:', streamError);
        if (streamError.name === 'AbortError') {
          if (aiMessageId) {
            updateStreamingMessage(aiMessageId, 'Agent plan generation stopped by user.');
          }
        } else if (streamError.name === 'TypeError' && streamError.message.includes('network')) {
          if (aiMessageId) {
            updateStreamingMessage(aiMessageId, '❌ Network error. Please check your connection and try again.');
          }
        } else {
          if (aiMessageId) {
                              updateStreamingMessage(aiMessageId, `❌ Agent plan generation error: ${streamError.message}`);
          }
        }
      } finally {
        reader.releaseLock();
        console.log(`🔒 Plan reader released, streaming loop completed`);
      }
      
    } catch (error: any) {
      console.error('Plan generation failed:', error);
      setError(error.message);
      if (aiMessageId) {
        updateStreamingMessage(aiMessageId, `❌ Failed to start agent plan generation: ${error.message}`);
      }
      setIsStreaming(false);
    } finally {
      console.log(`🏁 Plan generation finished, cleaning up...`);
      setIsGenerating(false);
      setIsStreaming(false);
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
      addMessage('ai', `Agent plan executed successfully! Check the results above.`, undefined, undefined, false);
      
      // Check for component tags in the execution result
      if (data.message) {
        console.log('🔍 Execution result message:', data.message);
        handleComponentTags(data.message);
      }
      
    } catch (err) {
      console.error("Error executing plan:", err);
      const errorMessage = err instanceof Error ? err.message : "Failed to execute agent plan";
      setError(errorMessage);
      
      // Update the message with error result
      setMessages(prev => prev.map(msg => 
        msg.plan && plan && JSON.stringify(msg.plan) === JSON.stringify(plan)
          ? { ...msg, isExecuting: false, executionResult: { error: errorMessage } }
          : msg
      ));
      
      // Check for component tags in the error message
      handleComponentTags(errorMessage);
    } finally {
      setIsExecuting(false);
    }
  };



  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isGenerating) return;

    const prompt = inputValue.trim();
    setInputValue("");
    
    // Choose between streaming chat and agent planning based on mode
    if (chatMode === 'streaming') {
      await streamingChat(prompt);
    } else {
      await generateAgentPlan(prompt);
    }
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
      addMessage('ai', `Successfully indexed ${data.indexed_files || 0} backend files for AI context.`, undefined, undefined, false);
    } catch (err) {
      addMessage('ai', `Failed to index files: ${err instanceof Error ? err.message : "Unknown error"}`, undefined, undefined, false);
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
          data-chat-dialog
          className="fixed bottom-20 left-1/2 transform -translate-x-1/2 z-40"
          onMouseEnter={() => {
            if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current);
            setIsHovered(true);
          }}
          onMouseLeave={() => {
            hoverTimeoutRef.current = setTimeout(() => setIsHovered(false), 2000);
          }}
        >
          <Card className="w-full max-w-6xl max-h-[70vh] shadow-lg border-2 border-primary/20 bg-background/95 backdrop-blur-sm">
            <div className="p-4">
              <div className="flex justify-between items-center mb-3 pb-2 border-b">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium">
                  {chatMode === 'plan' ? '⚡ AI Flow Agent' : '💬 AI Chat Assistant'}
                </span>
                  <div className="flex items-center gap-1 bg-muted rounded-lg p-1">
                    <Button
                      size="sm"
                      variant={chatMode === 'plan' ? 'default' : 'ghost'}
                      onClick={() => setChatMode('plan')}
                      className="h-6 px-2 text-xs"
                    >
                      <Zap className="h-3 w-3 mr-1" />
                      Agent
                    </Button>
                    <Button
                      size="sm"
                      variant={chatMode === 'streaming' ? 'default' : 'ghost'}
                      onClick={() => setChatMode('streaming')}
                      className="h-6 px-2 text-xs"
                    >
                      <MessageCircle className="h-3 w-3 mr-1" />
                      Chat
                    </Button>
                  </div>
                  {chatMode === 'plan' && isStreaming && (
                    <div className="flex items-center gap-2 text-xs text-blue-600 bg-blue-50 px-2 py-1 rounded">
                      <Loader2 className="h-3 w-3 animate-spin" />
                      <span>Agent Planning: Step {currentStep} of 7</span>
                      {currentStep > 0 && (
                        <span className="text-blue-500">
                          ({Object.keys(stepResults).filter(key => stepResults[key]?.status === 'completed').length} completed)
                        </span>
                      )}
                    </div>
                  )}
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={indexBackendFiles}
                  className="h-6 px-2 text-xs"
                >
                  <Database className="h-3 w-3 mr-1" />
                  Index Files
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    console.log('🔍 Templates object:', templates);
                    console.log('🔍 Available template keys:', Object.keys(templates));
                    console.log('🔍 Sample template (first 3):', Object.keys(templates).slice(0, 3).map(key => ({
                      key,
                      display_name: templates[key]?.display_name,
                      description: templates[key]?.description
                    })));
                  }}
                  className="h-6 px-2 text-xs"
                >
                  <MessageSquare className="h-3 w-3 mr-1" />
                  Debug Templates
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    const testText = 'Here is a <ChatInput> component and a <ChatOutput> component';
                    console.log('🧪 Testing component tag extraction...');
                    console.log('🧪 Test text:', testText);
                    const extracted = extractComponentTags(testText);
                    console.log('🧪 Extracted tags:', extracted);
                    
                    // Test with actual template keys
                    const firstTemplateKey = Object.keys(templates)[0];
                    if (firstTemplateKey) {
                      const testWithRealComponent = `Add a <${firstTemplateKey}> component`;
                      console.log('🧪 Test with real component:', testWithRealComponent);
                      const extractedReal = extractComponentTags(testWithRealComponent);
                      console.log('🧪 Extracted real component:', extractedReal);
                    }
                  }}
                  className="h-6 px-2 text-xs"
                >
                  <Play className="h-3 w-3 mr-1" />
                  Test Tags
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    console.log('🧪 Testing component addition...');
                    const firstTemplateKey = Object.keys(templates)[0];
                    if (firstTemplateKey && flowId) {
                      try {
                        console.log(`🧪 Adding component: ${firstTemplateKey}`);
                        console.log(`🧪 Template:`, templates[firstTemplateKey]);
                        addComponent(templates[firstTemplateKey], firstTemplateKey);
                        console.log(`✅ Test: Added component ${firstTemplateKey}`);
                      } catch (error) {
                        console.error(`❌ Test: Failed to add component ${firstTemplateKey}:`, error);
                      }
                    } else {
                      console.warn('🧪 Test: No components or flow available for testing');
                    }
                  }}
                  className="h-6 px-2 text-xs"
                >
                  <CheckCircle className="h-3 w-3 mr-1" />
                  Test Add
                </Button>
              </div>
              <div ref={chatContainerRef} className="overflow-y-auto max-h-[50vh]">
                <div className="space-y-3">
                {!Array.isArray(messages) || messages.length === 0 ? (
                  <div className="text-center text-muted-foreground py-8">
                    {chatMode === 'plan' ? (
                      <>
                        <Brain className="h-12 w-12 mx-auto mb-3 opacity-50" />
                        <p>Start a conversation with your AI Flow Agent</p>
                        <p className="text-sm mt-1">Try asking: "Add error handling to my flow"</p>
                      </>
                    ) : (
                      <>
                        <MessageCircle className="h-12 w-12 mx-auto mb-3 opacity-50" />
                        <p>Start a conversation with your AI assistant</p>
                        <p className="text-sm mt-1">Try asking: "What does this flow do?" or "How can I improve this?"</p>
                      </>
                    )}
                  </div>
                ) : (
                  messages.filter(msg => msg && typeof msg === 'object' && msg.id).map((message, index) => {
                    // Ensure we have a unique key for each message with additional uniqueness
                    const uniqueKey = `${message.id}-${index}-${message.timestamp?.getTime() || Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
                    return (
                    <div key={uniqueKey} className={`flex gap-3 ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
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
                        <div className="text-sm">
                          {message.isStreaming ? (
                            <>
                              {message.streamedContent || ''}
                              <span className="inline-block w-0.5 h-4 bg-current animate-pulse ml-0.5 opacity-100"></span>
                            </>
                          ) : (
                            <>
                              {/* Try to format as structured response first, fallback to safeRender */}
                              {(() => {
                                // First check if we have llm_response data to display
                                if (message.llm_response) {
                                  return (
                                    <div>
                                      <div className="mb-2 text-xs text-green-600 bg-green-50 px-2 py-1 rounded border border-green-200 inline-flex items-center gap-1">
                                        <CheckCircle className="h-3 w-3" />
                                        Structured Response Detected
                                      </div>
                                      {formatResponseForDisplay(message.llm_response)}
                                    </div>
                                  );
                                }
                                
                                // Then check if content looks like JSON
                                try {
                                  if (message.content && typeof message.content === 'string' && 
                                      (message.content.trim().startsWith('{') || message.content.trim().startsWith('['))) {
                                    const parsed = JSON.parse(message.content);
                                    return (
                                      <div>
                                        <div className="mb-2 text-xs text-green-600 bg-green-50 px-2 py-1 rounded border border-green-200 inline-flex items-center gap-1">
                                          <CheckCircle className="h-3 w-3" />
                                          Structured Response Detected
                                        </div>
                                        {formatResponseForDisplay(parsed)}
                                      </div>
                                    );
                                  }
                                  // If not JSON, use safeRender
                                  return safeRender(message.content);
                                } catch {
                                  // If parsing fails, use safeRender
                                  return safeRender(message.content);
                                }
                              })()}
                              {message.content && message.content.includes('❌') && (
                                <div className="mt-2">
                                  <Button
                                    size="sm"
                                    variant="outline"
                                    onClick={() => {
                                      const lastUserMessage = messages.slice().reverse().find(msg => msg.type === 'user');
                                      if (lastUserMessage) {
                                        streamingChat(lastUserMessage.content);
                                      }
                                    }}
                                    className="h-6 px-2 text-xs"
                                  >
                                    <RefreshCw className="h-3 w-3 mr-1" />
                                    Retry
                                  </Button>
                                </div>
                              )}
                            </>
                          )}
                        </div>
                        {message.plan && (
                          <div className="mt-3 p-3 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg" style={{display: 'none'}}>
                            
                            
                           
                            
                            {/* Component Tags Detected */}
                            {message.plan.message && message.plan.message.includes('<') && message.plan.message.includes('>') && (
                              <div className="mt-3 p-2 bg-green-50 border border-green-200 rounded">
                                <div className="text-xs text-green-700">
                                  <strong>✅ Component tags detected!</strong> The addComponent.js should automatically add these to your canvas.
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

                            {/* Show streaming progress if this is the streaming message */}
                            {isStreaming && message.type === 'ai' && message.content.includes('Starting to create a 7-step agent plan') && (
                              <div className="mt-3 space-y-2">
                                <div className="text-xs font-medium text-blue-800 mb-2">Agent Planning Progress (Step {currentStep} of 7):</div>
                                {[
                                  {num: 1, name: 'Strategic Planning', key: 'strategic_planning'},
                                  {num: 2, name: 'Objective Definition', key: 'objective'},
                                  {num: 3, name: 'Current State Analysis', key: 'current_state_analysis'},
                                  {num: 4, name: 'Required Changes', key: 'required_changes'},
                                  {num: 5, name: 'Execution Strategy', key: 'execution_strategy'},
                                  {num: 6, name: 'Step Design', key: 'step_design'},
                                  {num: 7, name: 'Component Quality Validation', key: 'component_quality_validation'}
                                ].map(({num, name, key}) => (
                                  <div key={num} className="space-y-1">
                                    <div className="flex items-center gap-2 text-xs">
                                      {stepResults[num]?.status === 'completed' ? (
                                        <CheckCircle className="h-3 w-3 text-green-500" />
                                      ) : stepResults[num]?.status === 'streaming' ? (
                                        <Loader2 className="h-3 w-3 text-blue-500 animate-spin" />
                                      ) : messages.some(msg => 
                                        msg.content.includes(`Step ${num}:`) && msg.isStreaming
                                      ) ? (
                                        <Loader2 className="h-3 w-3 text-green-500 animate-spin" />
                                      ) : stepResults[num]?.status === 'starting' ? (
                                        <Loader2 className="h-3 w-3 text-blue-500 animate-spin" />
                                      ) : (
                                        <div className="h-3 w-3 rounded-full border-2 border-gray-300" />
                                      )}
                                      <span className={
                                        stepResults[num]?.status === 'completed' ? 'text-green-700' : 
                                        stepResults[num]?.status === 'streaming' ? 'text-blue-700' : 
                                        messages.some(msg => msg.content.includes(`Step ${num}:`) && msg.isStreaming) ? 'text-green-700' :
                                        stepResults[num]?.status === 'starting' ? 'text-blue-700' : 
                                        'text-gray-500'
                                      }>
                                        Step {num}: {name}
                                        {stepResults[num]?.status === 'completed' && ' ✓'}
                                        {stepResults[num]?.status === 'streaming' && ' 🔄'}
                                        {messages.some(msg => msg.content.includes(`Step ${num}:`) && msg.isStreaming) && ' 🔄'}
                                        {stepResults[num]?.status === 'starting' && '...'}
                                      </span>
                                    </div>
                                    
                                    {/* Show LLM output for completed steps */}
                                    {stepResults[num]?.status === 'completed' && stepResults[num]?.llm_output && (
                                      <div className="ml-5 text-xs text-gray-600 bg-gray-50 p-2 rounded border-l-2 border-gray-300">
                                        <div className="font-medium mb-1">LLM Output:</div>
                                        <div className="text-xs">
                                          {formatResponseForDisplay(stepResults[num]?.llm_output)}
                                        </div>
                                      </div>
                                    )}
                                    
                                    {/* Show streaming content in real-time */}
                                    {stepResults[num]?.status === 'streaming' && stepResults[num]?.streamed_content && (
                                      <div className="ml-5 text-xs text-blue-600 bg-blue-50 p-2 rounded border-l-2 border-blue-300">
                                        <div className="font-medium mb-1">Streaming Response:</div>
                                        <pre className="whitespace-pre-wrap text-xs">
                                          {stepResults[num].streamed_content}
                                          <span className="inline-block w-0.5 h-4 bg-current animate-pulse ml-0.5 opacity-100"></span>
                                        </pre>
                                      </div>
                                    )}
                                    
                                    {/* Show completed LLM output */}
                                    {stepResults[num]?.status === 'completed' && stepResults[num]?.llm_output && (
                                      <div className="ml-5 text-xs text-gray-600 bg-gray-50 p-2 rounded border-l-2 border-gray-300">
                                        <div className="font-medium mb-1">LLM Output:</div>
                                        <div className="text-xs">
                                          {formatResponseForDisplay(stepResults[num]?.llm_output)}
                                        </div>
                                      </div>
                                    )}
                                    
                                    {/* Show streaming content from messages */}
                                    {messages.some(msg => 
                                      msg.content.includes(`Step ${num}:`) && msg.isStreaming && msg.streamedContent
                                    ) && (
                                      <div className="ml-5 text-xs text-green-600 bg-green-50 p-2 rounded border-l-2 border-green-300">
                                        <div className="font-medium mb-1">Live Streaming:</div>
                                        <pre className="whitespace-pre-wrap text-xs">
                                          {messages.find(msg => 
                                            msg.content.includes(`Step ${num}:`) && msg.isStreaming
                                          )?.streamedContent || ''}
                                          <span className="inline-block w-0.5 h-4 bg-current animate-pulse ml-0.5 opacity-100"></span>
                                        </pre>
                                      </div>
                                    )}
                                  </div>
                                ))}
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
                  );
                  })
                )}
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Floating Chat Input Bar */}
      <div 
        data-chat-input
        className="fixed bottom-6 left-1/2 transform -translate-x-1/2 z-50"
        onMouseEnter={() => {
          if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current);
          setIsHovered(true);
        }}
        onMouseLeave={() => {
          hoverTimeoutRef.current = setTimeout(() => setIsHovered(false), 2000);
        }}
      >
        <Card className="shadow-lg border-2 border-primary/20 bg-background/95 backdrop-blur-sm">
          <CardContent className="p-0">
            <form onSubmit={handleSubmit} className="flex items-center gap-2 p-2">
              <div className="flex items-center gap-2 text-sm text-muted-foreground px-2">
                {chatMode === 'plan' ? (
                  <Brain className="h-4 w-4" />
                ) : (
                  <MessageCircle className="h-4 w-4" />
                )}
                <span className="hidden sm:inline">
                  {!flowId ? "No Flow Loaded" : chatMode === 'plan' ? "AI Agent Mode" : "AI Chat Mode"}
                </span>
                {chatMode === 'plan' && (
                  <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">
                    Will generate 7-step agent plan
                  </span>
                )}
              </div>
              
              <Textarea
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={!flowId ? "No flow loaded..." : chatMode === 'plan' ? "Ask your AI agent to modify your flow..." : "Chat with me about your flow..."}
                className="min-w-[500px] max-w-[800px] resize-none border-0 focus-visible:ring-0 focus-visible:ring-offset-0 bg-transparent"
                rows={1}
                disabled={isGenerating || !flowId}
              />
              
              <div className="flex items-center gap-1">
                {isGenerating && abortController && (
                  <Button
                    type="button"
                    size="sm"
                    variant="destructive"
                    onClick={stopStreaming}
                    className="h-8 w-8 p-0"
                    title="Stop streaming"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                )}
                <Button
                    type="submit"
                    size="sm"
                    disabled={isGenerating || !inputValue.trim() || !flowId}
                    className="h-8 w-8 p-0"
                  >
                  {isGenerating ? (
                    isStreaming ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    )
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

