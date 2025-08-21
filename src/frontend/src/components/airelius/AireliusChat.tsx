import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent } from "@/components/ui/card";
import { Loader2, Send, X, Brain, Play, CheckCircle, XCircle, Upload, Database, RefreshCw, MessageSquare, MessageCircle, Zap, Plus, Info } from "lucide-react";
import useFlowStore from "@/stores/flowStore";
import useFlowsManagerStore from "@/stores/flowsManagerStore";
import { useTypesStore } from "@/stores/typesStore";
import { useAddComponent } from "@/hooks/use-add-component";
import { scapedJSONStringfy } from "@/utils/reactflowUtils";
import { edgeParser } from "./EdgeParser";

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
 * 
 * Component Tag Format for LLM:
 * The LLM can specify components to add using special tags with the following format:
 * 
 * 1. Single Component (no connections):
 *    <ComponentName-GeneratedId>
 *    Example: <ChatInput-abc123>
 * 
 * 2. Single Connection:
 *    <ComponentName-GeneratedId:ExistingComponentId>
 *    Example: <WebSearchNoAPI-xyz789:ChatInput-abc123>
 * 
 * 3. Multiple Connections (comma-separated):
 *    <ComponentName-GeneratedId:ExistingComponentId1,ExistingComponentId2,ExistingComponentId3>
 *    Example: <WebSearchNoAPI-xyz789:ChatInput-abc123,PromptTemplate-def456,MemoryStore-ghi789>
 * 
 * The system will automatically:
 * - Create the specified component
 * - Establish connections to all existing components listed after the colon
 * - Handle edge creation with proper Langflow structure
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
  const componentData = useTypesStore((state) => state.data);
  const addComponent = useAddComponent();
  
  // Get access to nodes for building instances map
  const nodes = useFlowStore((state) => state.nodes);

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

  // Function to create edges between components by name (legacy)
  const createEdge = (sourceComponentName: string, targetComponentName: string) => {
    // Get the flow store to add edges
    const { setEdges, nodes } = useFlowStore.getState();
    
    // Find the actual nodes by component name
    const sourceNode = nodes.find(node => {
      if (node.data?.node && 'display_name' in node.data.node) {
        return node.data.node.display_name === sourceComponentName;
      }
      return false;
    });
    
    const targetNode = nodes.find(node => {
      if (node.data?.node && 'display_name' in node.data.node) {
        return node.data.node.display_name === targetComponentName;
      }
      return false;
    });
    
    if (!sourceNode || !targetNode) {
      //console.warn(`⚠️ Cannot create edge: ${sourceComponentName} -> ${targetComponentName}`);
      //console.warn(`   Source node found: ${!!sourceNode}, Target node found: ${!!targetNode}`);
      return;
    }
    
    // Create proper handle objects
    const sourceHandle = {
      baseClasses: [],
      dataType: "string",
      id: sourceNode.id,
      output_types: ["string"],
      name: "output"
    };
    
    const targetHandle = {
      inputTypes: ["string"],
      output_types: [],
      type: "string",
      fieldName: "input",
      name: "input",
      id: targetNode.id
    };
    
    // Convert to JSON strings using the proper function
    const sourceHandleString = scapedJSONStringfy(sourceHandle);
    const targetHandleString = scapedJSONStringfy(targetHandle);
    
    // Create edge with proper handles
    const newEdge = {
      id: `edge-${Date.now()}`,
      source: sourceNode.id,
      target: targetNode.id,
      sourceHandle: sourceHandleString,
      targetHandle: targetHandleString
    };
    
    // Add edge to flow store
    setEdges((oldEdges) => [...oldEdges, newEdge]);
    //console.log(`🔗 Created edge: ${sourceComponentName}(${sourceNode.id}) -> ${targetComponentName}(${targetNode.id})`, newEdge);
  };

  // Function to create edges between components using their actual node IDs
  const createEdgeByIds = (sourceNodeId: string, targetNodeId: string) => {
    // Get the current nodes directly from the flow store to avoid timing issues
    const currentNodes = useFlowStore.getState().nodes;
    //console.log(`🔍 Looking for nodes: source=${sourceNodeId}, target=${targetNodeId}`);
    //console.log(`🔍 Available nodes from store:`, currentNodes.map(n => ({ id: n.id, type: n.data?.type })));
    
    const sourceNode = currentNodes.find(node => node.id === sourceNodeId);
    const targetNode = currentNodes.find(node => node.id === targetNodeId);
    
    //console.log(`🔍 Source node found:`, sourceNode ? { id: sourceNode.id, type: sourceNode.data?.type } : 'NOT FOUND');
    //console.log(`🔍 Target node found:`, targetNode ? { id: targetNode.id, type: targetNode.data?.type } : 'NOT FOUND');
    
    if (sourceNode && targetNode) {
      // Use the new handle selection algorithm
      const handleSelection = selectHandles(sourceNodeId, targetNodeId);
      //console.log(`🔍 Handle selection result:`, handleSelection);
      
      if (handleSelection.needsAdapter) {
        //console.log(`🔧 Adapter needed: ${handleSelection.adapterType}`);
        // TODO: Implement automatic adapter insertion
        //console.warn(`⚠️ Automatic adapter insertion not yet implemented. Need to add ${handleSelection.adapterType} between ${sourceNodeId} and ${targetNodeId}`);
        return;
      }
      
      if (!handleSelection.sourceHandle || !handleSelection.targetHandle) {
        //console.warn(`⚠️ Could not select compatible handles for edge creation`);
        return;
      }
      
      // Create edge with proper Langflow structure using selected handles
      const sourceHandle = {
        dataType: sourceNode.data?.type || "string",
        id: sourceNodeId,
        name: handleSelection.sourceHandle.name,
        output_types: handleSelection.sourceHandle.output_types
      };
      
      const targetHandle = {
        fieldName: handleSelection.targetHandle.fieldName,
        id: targetNodeId,
        inputTypes: handleSelection.targetHandle.inputTypes,
        type: normalizeType(handleSelection.targetHandle.inputTypes[0] || "str")
      };
      
      // Convert to JSON strings using the broken but required function
      const sourceHandleString = scapedJSONStringfy(sourceHandle);
      const targetHandleString = scapedJSONStringfy(targetHandle);
      
      // Create edge with proper structure
      const edgeId = `xy-edge__${sourceNodeId}${sourceHandleString}-${targetNodeId}${targetHandleString}`;
      
      // Check for potential edge ID conflicts
      const existingEdges = useFlowStore.getState().edges;
      const conflictingEdge = existingEdges.find(e => e.id === edgeId);
      if (conflictingEdge) {
        //console.warn(`⚠️ EDGE ID CONFLICT DETECTED!`);
        //console.warn(`   New edge ID: ${edgeId}`);
        //console.warn(`   Conflicting edge:`, conflictingEdge);
        //console.warn(`   This might cause the new edge to overwrite the existing one!`);
      }
      
      const newEdge = {
        id: edgeId,
        source: sourceNodeId,
        target: targetNodeId,
        sourceHandle: sourceHandleString,
        targetHandle: targetHandleString,
        data: {
          sourceHandle: sourceHandle,
          targetHandle: targetHandle
        }
      };
      
      // Add edge to flow store with detailed debugging
      //console.log(`🔍 BEFORE adding edge - Current edges count:`, useFlowStore.getState().edges.length);
      //console.log(`🔍 New edge to add:`, newEdge);
      
      setEdges((oldEdges) => {
        const newEdges = [...oldEdges, newEdge];
        //console.log(`🔍 AFTER adding edge - New edges count: ${newEdges.length}`);
        //console.log(`🔍 Edge ID being added: ${newEdge.id}`);
        //console.log(`🔍 All edges after addition:`, newEdges.map(e => ({ id: e.id, source: e.source, target: e.target })));
        return newEdges;
      });
      
      //console.log(`🔗 Created edge by IDs with proper structure: ${sourceNodeId} -> ${targetNodeId}`);
      
      // Verify edge was actually added
      setTimeout(() => {
        const currentEdges = useFlowStore.getState().edges;
        const edgeExists = currentEdges.some(e => e.id === newEdge.id);
        //console.log(`🔍 VERIFICATION - Edge ${newEdge.id} exists in store: ${edgeExists}`);
        //console.log(`🔍 Total edges in store: ${currentEdges.length}`);
        if (!edgeExists) {
          //console.error(`❌ EDGE CREATION FAILED - Edge ${newEdge.id} was not added to store!`);
        }
      }, 100);
    } else {
      //console.warn(`⚠️ Cannot create edge: source or target node not found`);
      //console.warn(`   Source node found: ${!!sourceNode}, Target node found: ${!!targetNode}`);
    }
  };

  // State to track pending edge connections
  const [pendingEdges, setPendingEdges] = useState<Array<{source: string, target: string}>>([]);
  
  // Get current nodes and setEdges from flow store to watch for changes
  const setEdges = useFlowStore((state) => state.setEdges);
  
  // Effect to create edges when components are added
  useEffect(() => {
    //console.log(`🔍 useEffect triggered - pendingEdges: ${pendingEdges.length}, nodes: ${nodes.length}`);
    
    if (pendingEdges.length === 0) {
      //console.log('🔍 No pending edges, returning early');
      return;
    }
    
    //console.log('🔍 Current pending edges:', pendingEdges);
    //console.log('🔍 Current nodes:', nodes.map(n => ({ 
      //id: n.id, 
      //type: n.data?.type, 
      //display_name: n.data?.node && 'display_name' in n.data.node ? n.data.node.display_name : 'N/A' 
    //})));
    
    // Try to create edges for pending connections
    const remainingEdges = pendingEdges.filter(edge => {
      //console.log(`🔍 Processing edge: ${edge.source} -> ${edge.target}`);
      
      const sourceNode = nodes.find(node => {
        if (node.data?.node && 'display_name' in node.data.node) {
          const matches = node.data.node.display_name === edge.source;
          //console.log(`🔍 Source node ${node.id}: display_name="${node.data.node.display_name}" matches "${edge.source}": ${matches}`);
          return matches;
        }
        //console.log(`🔍 Source node ${node.id}: no display_name property`);
        return false;
      });
      
      const targetNode = nodes.find(node => {
        if (node.data?.node && 'display_name' in node.data.node) {
          const matches = node.data.node.display_name === edge.target;
          //console.log(`🔍 Target node ${node.id}: display_name="${node.data.node.display_name}" matches "${edge.target}": ${matches}`);
          return matches;
        }
        //console.log(`🔍 Target node ${node.id}: no display_name property`);
        return false;
      });
      
      if (sourceNode && targetNode) {
        //console.log(`🔍 Both nodes found! Creating edge: ${edge.source} -> ${edge.target}`);
        // Both nodes found, create the edge
        createEdge(edge.source, edge.target);
        //console.log(`🔗 Successfully created edge: ${edge.source} -> ${edge.target}`);
        return false; // Remove from pending
      } else {
        //console.log(`🔍 Nodes not found yet. Source: ${!!sourceNode}, Target: ${!!targetNode}`);
      }
      
      return true; // Keep in pending
    });
    
    if (remainingEdges.length !== pendingEdges.length) {
      setPendingEdges(remainingEdges);
      //console.log(`🔗 Processed edges, ${pendingEdges.length - remainingEdges.length} edges created, ${remainingEdges.length} remaining`);
    } else {
      //console.log(`🔍 No edges processed, all ${pendingEdges.length} edges still pending`);
    }
  }, [nodes, pendingEdges]);
  
  // Function to add components when LLM outputs component tags
  const handleComponentTags = async (llmOutput: string) => {
    if (!flowId || !currentFlow) {
      //console.warn('No flow loaded, cannot add components');
      return;
    }
    
    if (Object.keys(templates).length === 0) {
      //console.warn('No templates loaded, cannot add components');
      return;
    }
    
    //console.log('🔍 LLM Output:', llmOutput);
    //console.log('🔍 Available templates:', Object.keys(templates));
    
    const componentTags = extractComponentTags(llmOutput);
    //console.log('🔍 Extracted component tags:', componentTags);
    
    for (const tag of componentTags) {
      //console.log(`🔍 Looking for component: "${tag.name}"`);
      //console.log(`🔍 Template exists:`, !!templates[tag.name]);
      
      if (templates[tag.name]) {
        try {
          // Add the component to the flow with the generated ID
          const fullComponentName = tag.generatedId; // Use just the generatedId to avoid duplicate prefixes
          const nodeId = addComponent(templates[tag.name], tag.name, undefined, fullComponentName);
          console.log(`✅ Added component: ${fullComponentName} with node ID: ${nodeId}`);
          console.log(`🔍 Component details:`, {
            name: tag.name,
            generatedId: tag.generatedId,
            fullComponentName: fullComponentName,
            nodeId: nodeId,
            template: templates[tag.name]
          });
          
          // Handle multiple connections using the parsed addSimpleEdgeCalls data
          if (tag.addSimpleEdgeCalls && tag.addSimpleEdgeCalls.length > 0) {
            console.log(`🔗 Creating ${tag.addSimpleEdgeCalls.length} connection(s) for ${fullComponentName} using parsed handle data`);
            console.log(`🔗 addSimpleEdgeCalls:`, tag.addSimpleEdgeCalls);
            
            // Start edge creation status reporting
            setEdgeCreationStatus(prev => ({
              ...prev,
              isCreating: true,
              currentOperation: `Creating ${tag.addSimpleEdgeCalls.length} connection(s) for ${tag.name}`,
              progress: 0,
              total: tag.addSimpleEdgeCalls.length
            }));
            
            // Track edge creation results
            const edgeResults: Array<{target: string, success: boolean, error?: string}> = [];
            
            // Process connections using the parsed handle data
            for (let index = 0; index < tag.addSimpleEdgeCalls.length; index++) {
              const edgeCall = tag.addSimpleEdgeCalls[index];
              
              // Update progress
              reportEdgeProgress(`Creating connection ${index + 1}/${tag.addSimpleEdgeCalls.length}`, index, tag.addSimpleEdgeCalls.length);
              
              //console.log(`🔍 Processing connection ${index + 1}/${tag.addSimpleEdgeCalls.length}:`, edgeCall);
              
              // Find the target component by its ID in the flow
              const targetNode = nodes.find(node => node.id === edgeCall.targetNodeId);
              
              if (targetNode) {
                //console.log(`✅ Target node found for connection ${index + 1}:`, { id: targetNode.id, type: targetNode.data?.type });
                
                try {
                  console.log(`🔗 Attempting to create edge ${index + 1}: ${edgeCall.sourceNodeId} -> ${edgeCall.targetNodeId}`);
                  // Create edge using the parsed handle data
                  const result = await createEdgeWithParsedHandles(edgeCall);
                  if (result.success) {
                    console.log(`✅ Created edge ${index + 1}: ${edgeCall.sourceNodeId} -> ${edgeCall.targetNodeId}`);
                    edgeResults.push({ target: edgeCall.targetNodeId, success: true });
                    reportEdgeResult(edgeCall.sourceNodeId, edgeCall.targetNodeId, true, `Connection ${index + 1} created successfully`);
                  } else {
                    console.error(`❌ Failed to create edge ${index + 1}: ${edgeCall.sourceNodeId} -> ${edgeCall.targetNodeId}`, result.error);
                    edgeResults.push({ target: edgeCall.targetNodeId, success: false, error: result.error });
                    reportEdgeResult(edgeCall.sourceNodeId, edgeCall.targetNodeId, false, `Connection ${index + 1} failed: ${result.error}`);
                  }
                } catch (error) {
                  console.error(`❌ Failed to create edge ${index + 1}: ${edgeCall.sourceNodeId} -> ${edgeCall.targetNodeId}`, error);
                  edgeResults.push({ target: edgeCall.targetNodeId, success: false, error: String(error) });
                  reportEdgeResult(edgeCall.sourceNodeId, edgeCall.targetNodeId, false, `Connection ${index + 1} failed: ${String(error)}`);
                }
              } else {
                //console.warn(`⚠️ Target component ${edgeCall.targetNodeId} not found in flow for connection ${index + 1}`);
                edgeResults.push({ target: edgeCall.targetNodeId, success: false, error: 'Target not found' });
                reportEdgeResult(edgeCall.sourceNodeId, edgeCall.targetNodeId, false, `Connection ${index + 1} failed: Target not found`);
              }
            }
            
            // Final progress update
            reportEdgeProgress(`Completed ${tag.addSimpleEdgeCalls.length} connection(s)`, tag.addSimpleEdgeCalls.length, tag.addSimpleEdgeCalls.length);
            
            // Log summary of all edge creation attempts
            //console.log(`📊 EDGE CREATION SUMMARY for ${fullComponentName}:`);
            edgeResults.forEach((result, index) => {
              const status = result.success ? '✅ SUCCESS' : '❌ FAILED';
              const error = result.error ? ` (Error: ${result.error})` : '';
              //console.log(`   Connection ${index + 1}: ${nodeId} -> ${result.target}: ${status}${error}`);
            });
            
            const successCount = edgeResults.filter(r => r.success).length;
            const failureCount = edgeResults.filter(r => !r.success).length;
            //console.log(`📊 Final Result: ${successCount} successful, ${failureCount} failed out of ${tag.addSimpleEdgeCalls.length} total connections`);
            
            // Auto-clear status after a delay
            setTimeout(() => {
              clearEdgeStatus();
            }, 5000);
            
          } else {
            //console.log(`ℹ️ No connections specified for ${fullComponentName}, component added without edges`);
          }
        } catch (error) {
          //console.error(`❌ Failed to add component ${tag.name}:`, error);
        }
      } else {
        //console.warn(`⚠️ Component type "${tag.name}" not found in templates`);
        // Try to find similar component names
        const similarComponents = Object.keys(templates).filter(name => 
          name.toLowerCase().includes(tag.name.toLowerCase()) || 
          tag.name.toLowerCase().includes(name.toLowerCase())
        );
        if (similarComponents.length > 0) {
          //console.log(`💡 Similar components found:`, similarComponents);
        }
      }
    }
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
          //console.log('Click outside detected, closing dialog');
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
                          {tag.connectToIds && tag.connectToIds.length > 0 
                            ? `${tag.name}-${tag.generatedId} → ${tag.connectToIds.join(', ')}`
                            : `${tag.name}-${tag.generatedId}`
                          }
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

  // Inject component registry and instances into EdgeParser before using it
  useEffect(() => {
    if (componentData && nodes.length > 0) {
      (async () => {

      

      
      // Convert component data to the format expected by EdgeParser
      // Prefer authoritative registry file if available; fall back to typesStore mapping
      let registry: { components: Array<{ type: string; inputs: Record<string, string[]>; outputs: Record<string, string[]> }> };

      // Try loading the authoritative registry JSON via dynamic import
      // Note: bundlers resolve this at build time; if it fails, we'll fall back below
      let jsonRegistry: any | null = null;
      try {
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore - allow dynamic json import outside src alias
        jsonRegistry = await import('../../../../../component_registry_v2.json');
      } catch (e) {
        jsonRegistry = null;
      }

      if (!jsonRegistry) {
        try {
          const resp = await fetch('/component_registry_v2.json');
          if (resp.ok) {
            jsonRegistry = await resp.json();
          }
        } catch (err) {
          jsonRegistry = null;
        }
      }

      if (jsonRegistry?.components && Array.isArray(jsonRegistry.components)) {
        registry = {
          components: jsonRegistry.components.map((comp: any) => ({
            // Prefer display_name for matching with EdgeParser types (e.g., "Language Model")
            type: (comp.display_name || comp.type || '').trim(),
            inputs: comp.inputs || {},
            outputs: comp.outputs || {}
          }))
        };
      } else {
        // Fallback: build from typesStore (may have incomplete outputs)
        // DEBUG: Let's see what the actual componentData structure looks like
        console.log('🔍 DEBUG: componentData structure:', JSON.stringify(componentData, null, 2));
        registry = {
          components: Object.keys(componentData).flatMap(category => 
            Object.keys(componentData[category]).map(kind => {
              const component = componentData[category][kind];

              // Extract inputs from template (the actual structure from API)
              const inputs: { [key: string]: string[] } = {};
              if (component.template) {
                Object.keys(component.template).forEach(fieldName => {
                  const field = component.template[fieldName];
                  if (field.input_types && Array.isArray(field.input_types)) {
                    inputs[fieldName] = field.input_types;
                  } else if (field.type) {
                    inputs[fieldName] = [field.type];
                  } else {
                    inputs[fieldName] = [];
                  }
                });
              }
              
              // Extract outputs from component.outputs array
              const outputs: { [key: string]: string[] } = {};
              if (component.outputs && Array.isArray(component.outputs)) {
                component.outputs.forEach((output: any) => {
                  if (output.name && output.output_types) {
                    outputs[output.name] = output.output_types;
                  }
                });
              }
              
              // Use the exact type from the registry to ensure proper matching
              const componentType = (component.type || component.display_name || kind).trim();
              
              return {
                type: componentType,
                inputs: inputs,
                outputs: outputs
              };
            })
          )
        };
      }
      

      
      // Debug: Check if we have the expected components (removed duplicate declarations)
      
      // Debug: Check all component types to see what we actually have
      console.log('🔍 All component types in registry:', registry.components.map(c => c.type));
      
      // Debug: Check if we have the specific components we need
      const promptComponent = registry.components.find(c => c.type === 'Prompt Template');
      const textInputComponent = registry.components.find(c => c.type === 'TextInput');
      console.log('🔍 Found Prompt Template component:', !!promptComponent);
      console.log('🔍 Found TextInput component:', !!textInputComponent);
      
      // Debug: Show all components that contain "Text" or "Input" to see what we actually have
      const textLikeComponents = registry.components.filter(c => 
        c.type.toLowerCase().includes('text') || c.type.toLowerCase().includes('input')
      );
      console.log('🔍 Text/Input-like components:', textLikeComponents.map(c => c.type));
      
      // Debug: Show the first few components to see the structure
      console.log('🔍 First 5 components:', registry.components.slice(0, 5).map(c => ({ type: c.type, inputs: Object.keys(c.inputs || {}), outputs: Object.keys(c.outputs || {}) })));
      
      // Build instances map from current nodes
      const instancesMap = new Map();
      nodes.forEach(node => {
        if (node.data?.node) {
          // Use node.id as the key and get the component type from various possible locations
          const componentType = (node.data.node as any)._type || (node.data.node as any).type || node.data.node.display_name;
          instancesMap.set(node.id, {
            id: node.id,
            type: componentType
          });
          //console.log('🔍 Added to instances map:', { nodeId: node.id, componentType });
          
          // Also log the full node data to see what we're working with
          //console.log('🔍 Full node data for', node.id, ':', JSON.stringify(node.data.node, null, 2));
        }
      });
      
      // Inject into EdgeParser
      edgeParser.setComponentRegistry(registry);
      edgeParser.setInstancesMap(instancesMap);
      
      console.log('🔍 EdgeParser injected with registry:', registry.components.length, 'components and', instancesMap.size, 'instances');
      
      // Test the EdgeParser to make sure it's working
      if (registry.components.length > 0) {
        const testEdge = edgeParser.parseEdgeDefinitions(
          "TestComponent.out:test -> TargetComponent.in:input", 
          "TestComponent", 
          "test-id"
        );
        console.log('🔍 EdgeParser test result:', testEdge);
      }
      })();
    }
  }, [componentData, nodes]);

  // Helper function to extract component tags using the new EdgeParser
  const extractComponentTags = (text: string): Array<{
    name: string;
    generatedId: string;
    connectToIds: string[];
    addSimpleEdgeCalls: Array<{
      sourceNodeId: string;
      targetNodeId: string;
      sourceHandle: string;
      targetHandle: string;
      sourceHandleData: {
        dataType: string;
        name: string;
        output_types: string[];
      };
      targetHandleData: {
        fieldName: string;
        inputTypes: string[];
        type: string;
      };
    }>;
  }> => {
    if (!text || typeof text !== 'string') {
      return [];
    }
    
    // Use EdgeParser to find and parse all component tags
    // We need to manually find the complete tag since regex can't handle nested > characters in edge definitions
    const componentRegex = /<(\w+):/g;
    const matches: Array<{
      name: string;
      generatedId: string;
      connectToIds: string[];
      addSimpleEdgeCalls: Array<{
        sourceNodeId: string;
        targetNodeId: string;
        sourceHandle: string;
        targetHandle: string;
        sourceHandleData: {
          dataType: string;
          name: string;
          output_types: string[];
        };
        targetHandleData: {
          fieldName: string;
          inputTypes: string[];
          type: string;
        };
      }>;
    }> = [];
    
    let match;
    
    while ((match = componentRegex.exec(text)) !== null) {
      // Find the complete component tag by looking for the next < or end of string
      const startIndex = match.index;
      const nextTagIndex = text.indexOf('<', startIndex + 1);
      const endIndex = nextTagIndex > -1 ? nextTagIndex : text.length;
      
      // Extract the complete tag content
      const fullTagContent = text.substring(startIndex, endIndex);
      console.log('🔍 Found complete tag:', fullTagContent);
      
      // Extract componentType and the rest of the content
      const componentType = match[1];
      const tagContent = fullTagContent.substring(componentType.length + 1); // +1 for ":"
      
      // Reconstruct the full tag for EdgeParser
      // The format should be: <ComponentType:ComponentId: EDGES>
      const componentTag = componentType + ':' + tagContent;
      
      console.log('🔍 DEBUG: reconstructed componentTag:', componentTag);
      
      // Use the new EdgeParser to parse the component tag
      const parsedComponent = edgeParser.parseComponentTag(`<${componentTag}>`);
      
      if (parsedComponent) {
        // Convert EdgeParser edges to the expected format
        const addSimpleEdgeCalls = parsedComponent.edges.map(edge => ({
          sourceNodeId: edge.source,
          targetNodeId: edge.target,
          sourceHandle: edge.sourceHandle,
          targetHandle: edge.targetHandle,
          sourceHandleData: edge.data?.sourceHandle,  // ✅ Get from data.sourceHandle
          targetHandleData: edge.data?.targetHandle   // ✅ Get from data.targetHandle
        }));
        
        // Extract connectToIds from edges
        const connectToIds = parsedComponent.edges.flatMap(edge => [
          edge.source,
          edge.target
        ]).filter(id => id !== parsedComponent.componentId);
        
        matches.push({
          name: parsedComponent.componentType,
          generatedId: parsedComponent.componentId,
          connectToIds,
          addSimpleEdgeCalls
        });
      }
    }
    
    // Filter out common HTML tags
    const filteredMatches = matches.filter(tag => 
      !['html', 'head', 'body', 'div', 'span', 'p', 'br', 'hr', 'img', 'a', 'ul', 'ol', 'li', 'table', 'tr', 'td', 'th'].includes(tag.name.toLowerCase())
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
    //console.error("Global error in AireliusChat:", error, errorInfo);
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
        //console.log(`🧹 Cleaned up ${prev.length - uniqueMessages.length} duplicate messages`);
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
        //console.warn(`⚠️ Duplicate message ID detected: ${newId}, regenerating...`);
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
      //console.log('🛑 Streaming stopped by user');
    }
  };

  const streamingChat = async (prompt: string) => {
    if (!prompt.trim() || !flowId) return;

    //console.log(`🚀 Starting streaming chat with prompt: "${prompt}"`);
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

      //console.log(`📡 API response received, status: ${response.status}`);
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }

      //console.log(`📖 Starting to read streaming response...`);

      let streamedContent = '';
      
      try {
        while (true) {
          // Check if streaming was aborted
          if (controller.signal.aborted) {
            //console.log('🛑 Streaming aborted by user');
            if (aiMessageId) {
              updateStreamingMessage(aiMessageId, 'Streaming stopped by user.');
            }
            break;
          }

          const { done, value } = await reader.read();
          if (done) {
            //console.log(`📖 Reader finished, done: ${done}`);
            // If we have content but didn't get a done signal, finalize the message
            if (streamedContent && aiMessageId) {
              //console.log(`⚠️ Reader finished unexpectedly, finalizing message with content: "${streamedContent}"`);
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
          ////console.log(`📦 Received ${lines.length} lines, total bytes: ${value?.length || 0}`);
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                ////console.log(`📦 Parsed streaming data:`, data);
                
                if (data.error) {
                  //console.error('Streaming error:', data.error);
                  setError(data.error);
                  // Update the streaming message to show the error
                  if (aiMessageId) {
                    updateStreamingMessage(aiMessageId, `❌ Error: ${data.error}`);
                  }
                  break;
                }
                
                if (data.content) {
                  streamedContent += data.content;
                  //console.log(`🔄 Streaming content: "${data.content}" -> Total: "${streamedContent}"`);
                  if (aiMessageId) {
                    updateStreamingMessage(aiMessageId, streamedContent);
                    // Small delay to make streaming visible (adjust this value for desired speed)
                    await new Promise(resolve => setTimeout(resolve, 5));
                  }
                } else if (data.content === '') {
                  // Handle empty content gracefully
                  //console.log(`📭 Received empty content chunk`);
                }
                
                if (data.done) {
                  //console.log(`✅ Streaming completed. Final content: "${streamedContent}"`);
                  // Mark message as no longer streaming and finalize content
                  if (aiMessageId) {
                    setMessages(prev => prev.map(msg => 
                      msg.id === aiMessageId 
                        ? { ...msg, isStreaming: false, content: streamedContent, streamedContent: streamedContent }
                        : msg
                    ));
                    
                    // Check for component tags and add components automatically
                    //console.log('🔍 Final streamed content:', streamedContent);
                    //console.log('🔍 Calling handleComponentTags with:', streamedContent);
                    await handleComponentTags(streamedContent);
                    
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
                //console.error('Failed to parse streaming data:', e, 'Raw line:', line);
                // Continue processing other lines even if one fails
              }
            } else if (line.trim()) {
              //console.log(`📝 Non-data line received: "${line}"`);
            }
          }
        }
      } catch (streamError: any) {
        //console.error('Error during streaming loop:', streamError);
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
        //console.log(`🔒 Reader released, streaming loop completed`);
        
        // Safety check: ensure the message is properly finalized
        if (aiMessageId && streamedContent) {
          setMessages(prev => prev.map(msg => 
            msg.id === aiMessageId 
              ? { ...msg, isStreaming: false, content: streamedContent, streamedContent: streamedContent }
              : msg
          ));
          
          // Check for component tags in the final content
          await handleComponentTags(streamedContent);
        }
      }
      
    } catch (error: any) {
      //console.error('Streaming chat failed:', error);
      setError(error.message);
      // Update the streaming message to show the error
      if (aiMessageId) {
        updateStreamingMessage(aiMessageId, `❌ Failed to start streaming: ${error.message}`);
      }
    } finally {
      //console.log(`🏁 Streaming chat finished, cleaning up...`);
      setIsGenerating(false);
    }
  };

  const generateAgentPlan = async (prompt: string) => {
    if (!prompt.trim() || !flowId) return;

    //console.log(`🚀 Starting AI agent planning with prompt: "${prompt}"`);
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
          available_templates: Object.keys(templates),
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      //console.log(`📡 Plan API response received, status: ${response.status}`);
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }

      //console.log(`📖 Starting to read plan streaming response...`);

      try {
        while (true) {
          // Check if streaming was aborted
          if (controller.signal.aborted) {
            //console.log('🛑 Plan generation aborted by user');
                    if (aiMessageId) {
          updateStreamingMessage(aiMessageId, 'Agent plan generation stopped by user.');
        }
            break;
          }

          const { done, value } = await reader.read();
          if (done) {
            //console.log(`📖 Plan reader finished, done: ${done}`);
            break;
          }
          
          const text = new TextDecoder().decode(value);
          const lines = text.split('\n');
          ////console.log(`📦 Received ${lines.length} lines, total bytes: ${value?.length || 0}`);
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                ////console.log(`📦 Parsed plan streaming data:`, data);
                
                if (data.error) {
                  //console.error('Plan streaming error:', data.error);
                  setError(data.error);
                  if (aiMessageId) {
                    updateStreamingMessage(aiMessageId, `❌ Error: ${data.error}`);
                  }
                  break;
                }
                
                if (data.status === 'starting') {
                  //console.log('🔄 Step starting:', data);
                  setCurrentStep(data.step);
                  setStepResults(prev => ({
                    ...prev,
                    [data.step]: { status: 'starting', message: data.message }
                  }));
                  
                  // Add a new message for each step starting
                  addMessage('ai', `Step ${data.step}: ${data.message}...`, undefined, undefined, false);
                  
                  //console.log(`Step ${data.step} starting: ${data.message}`);
                }
                
                // Handle streaming content for each step
                if (data.type === 'streaming' && data.content) {
                  ////console.log(`🔄 Streaming content for step ${data.step}:`, data.content);
                  
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
                  //console.log('✅ Step completed:', data);
                  //console.log('🔍 Raw data.result:', data.result);
                  //console.log('🔍 data.result type:', typeof data.result);
                  //console.log('🔍 data.result keys:', data.result && typeof data.result === 'object' ? Object.keys(data.result) : 'Not an object');
                  
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
                  //console.log('🔍 Final llmOutput:', llmOutput);
                  
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
                    //console.log('🔍 Updated stepResults:', newStepResults);
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
                    //console.log('🔍 LLM output for step:', llmOutput);
                    await handleComponentTags(llmOutput);
                  
                  //console.log(`Step ${data.step} completed: ${data.message}`, data.result);
                  //console.log(`LLM Output extracted:`, llmOutput);
                }
                
                if (data.status === 'completed' && data.final_plan) {
                  //console.log('🎯 Final plan received:', data);
                  setFinalPlan(data.final_plan);
                  setIsStreaming(false);
                  
                  // Add the AI's direct response
                  addMessage('ai', data.message, data, undefined, false);
                  
                                      // Check for component tags in the AI's response
                    if (data.message) {
                      //console.log('🔍 AI response:', data.message);
                      await handleComponentTags(data.message);
                    }
                  
                  //console.log('AI response received:', data.message);
                  break;
                }
                
              } catch (e) {
                //console.error('Failed to parse plan streaming data:', e, 'Raw line:', line);
                // Continue processing other lines even if one fails
              }
            } else if (line.trim()) {
              //console.log(`📝 Non-data line received: "${line}"`);
            }
          }
        }
      } catch (streamError: any) {
        //console.error('Error during plan streaming loop:', streamError);
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
        //console.log(`🔒 Plan reader released, streaming loop completed`);
      }
      
    } catch (error: any) {
      //console.error('Plan generation failed:', error);
      setError(error.message);
      if (aiMessageId) {
        updateStreamingMessage(aiMessageId, `❌ Failed to start agent plan generation: ${error.message}`);
      }
      setIsStreaming(false);
    } finally {
      //console.log(`🏁 Plan generation finished, cleaning up...`);
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
      //console.log("=== EXECUTING PFU PLAN - API CALL DETAILS ===");
      //console.log("URL: /api/v1/airelius/pfu/execute");
      //console.log("Method: POST");
      //console.log("Flow ID:", flowId);
      //console.log("Max Steps: 10");
      //console.log("Request Body:");
      //console.log(JSON.stringify({
        //plan: plan,
        //flow_id: flowId,
        //max_steps: 10,
      //}, null, 2));
      //console.log("=== END API CALL DETAILS ===");
      
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
        //console.log("Received execution response:", data); // Debug logging
      } catch (jsonError) {
        //console.error("JSON parse error in execution:", jsonError); // Debug logging
        throw new Error("Invalid JSON response from server");
      }

      // Validate response structure
      if (!data || typeof data !== 'object') {
        //console.error("Invalid execution response structure:", data); // Debug logging
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
        //console.log('🔍 Execution result message:', data.message);
        await handleComponentTags(data.message);
      }
      
    } catch (err) {
      //console.error("Error executing plan:", err);
      const errorMessage = err instanceof Error ? err.message : "Failed to execute agent plan";
      setError(errorMessage);
      
      // Update the message with error result
      setMessages(prev => prev.map(msg => 
        msg.plan && plan && JSON.stringify(msg.plan) === JSON.stringify(plan)
          ? { ...msg, isExecuting: false, executionResult: { error: errorMessage } }
          : msg
      ));
      
      // Check for component tags in the error message
      await handleComponentTags(errorMessage);
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

  // Type compatibility system based on PFU spec
  type NormalizedType = "str" | "Message" | "Document" | "List[str]" | "List[Document]";

  // Type normalization function
  const normalizeType = (type: string): NormalizedType => {
    const t = type.toLowerCase();
    if (["str", "string", "text", "textinput"].includes(t)) return "str";
    if (["message", "chatmessage"].includes(t)) return "Message";
    if (["document", "doc", "langchaindocument"].includes(t)) return "Document";
    if (["list[str]", "stringlist", "texts"].includes(t)) return "List[str]";
    if (["list[document]", "documents", "docs"].includes(t)) return "List[Document]";
    return type as NormalizedType; // fallback: treat as-is
  };

  // Type compatibility checking
  const isCompatible = (sourceType: NormalizedType, targetType: NormalizedType): boolean => {
    if (sourceType === targetType) return true;
    
    // Direct easy wins
    if (sourceType === "str" && targetType === "Message") return true;
    if (sourceType === "Message" && targetType === "str") return true;
    
    // For now, return false for other mismatches (adapters will handle them)
    return false;
  };

  // Find adapters in the catalog
  const findAdapters = (fromType: NormalizedType, toType: NormalizedType): string[] => {
    const adapters: string[] = [];
    
    // Common single-hop adapters
    if (fromType === "Message" && toType === "str") {
      // Look for components that convert Message to str
      Object.keys(templates).forEach(name => {
        const template = templates[name];
        if (template.template) {
          // Check if this component has Message input and str output
          const hasMessageInput = Object.values(template.template).some((field: any) => 
            field.input_types && field.input_types.some((t: string) => normalizeType(t) === "Message")
          );
          const hasStrOutput = template.outputs && template.outputs.some((output: any) => 
            output.output_types && output.output_types.some((t: string) => normalizeType(t) === "str")
          );
          
          if (hasMessageInput && hasStrOutput) {
            adapters.push(name);
          }
        }
      });
    }
    
    if (fromType === "str" && toType === "Message") {
      // Look for components that convert str to Message
      Object.keys(templates).forEach(name => {
        const template = templates[name];
        if (template.template) {
          const hasStrInput = Object.values(template.template).some((field: any) => 
            field.input_types && field.input_types.some((t: string) => normalizeType(t) === "str")
          );
          const hasMessageOutput = template.outputs && template.outputs.some((output: any) => 
            output.output_types && output.output_types.some((t: string) => normalizeType(t) === "Message")
          );
          
          if (hasStrInput && hasMessageOutput) {
            adapters.push(name);
          }
        }
      });
    }
    
    // Additional adapter types
    if (fromType === "str" && toType === "List[str]") {
      // Look for text splitters
      Object.keys(templates).forEach(name => {
        const template = templates[name];
        if (template.template && name.toLowerCase().includes('split')) {
          const hasStrInput = Object.values(template.template).some((field: any) => 
            field.input_types && field.input_types.some((t: string) => normalizeType(t) === "str")
          );
          const hasListStrOutput = template.outputs && template.outputs.some((output: any) => 
            output.output_types && output.output_types.some((t: string) => normalizeType(t) === "List[str]")
          );
          
          if (hasStrInput && hasListStrOutput) {
            adapters.push(name);
          }
        }
      });
    }
    
    if (fromType === "str" && toType === "Document") {
      // Look for document creators
      Object.keys(templates).forEach(name => {
        const template = templates[name];
        if (template.template && (name.toLowerCase().includes('document') || name.toLowerCase().includes('loader'))) {
          const hasStrInput = Object.values(template.template).some((field: any) => 
            field.input_types && field.input_types.some((t: string) => normalizeType(t) === "str")
          );
          const hasDocumentOutput = template.outputs && template.outputs.some((output: any) => 
            output.output_types && output.output_types.some((t: string) => normalizeType(t) === "Document")
          );
          
          if (hasStrInput && hasDocumentOutput) {
            adapters.push(name);
          }
        }
      });
    }
    
    if (fromType === "List[Document]" && toType === "Document") {
      // Look for document selectors/mergers
      Object.keys(templates).forEach(name => {
        const template = templates[name];
        if (template.template && (name.toLowerCase().includes('select') || name.toLowerCase().includes('merge'))) {
          const hasListDocInput = Object.values(template.template).some((field: any) => 
            field.input_types && field.input_types.some((t: string) => normalizeType(t) === "List[Document]")
          );
          const hasDocumentOutput = template.outputs && template.outputs.some((output: any) => 
            output.output_types && output.output_types.some((t: string) => normalizeType(t) === "Document")
          );
          
          if (hasListDocInput && hasDocumentOutput) {
            adapters.push(name);
          }
        }
      });
    }
    
    return adapters;
  };

  // Get component outputs
  const getComponentOutputs = (nodeId: string): Array<{name: string, output_types: string[]}> => {
    const node = nodes.find(n => n.id === nodeId);
    if (!node || !node.data?.node?.outputs) return [];
    
    return node.data.node.outputs.map((output: any) => ({
      name: output.name,
      output_types: output.output_types || []
    }));
  };

  // Get component inputs
  const getComponentInputs = (nodeId: string): Array<{fieldName: string, inputTypes: string[]}> => {
    const node = nodes.find(n => n.id === nodeId);
    if (!node || !node.data?.node?.template) return [];
    
    const inputs: Array<{fieldName: string, inputTypes: string[]}> = [];
    
    Object.entries(node.data.node.template).forEach(([fieldName, field]: [string, any]) => {
      if (field.input_types) {
        inputs.push({
          fieldName,
          inputTypes: field.input_types
        });
      }
    });
    
    return inputs;
  };

  // Handle selection algorithm based on PFU spec
  const selectHandles = (sourceNodeId: string, targetNodeId: string): {
    sourceHandle: { name: string, output_types: string[], id: string } | null,
    targetHandle: { fieldName: string, inputTypes: string[], id: string } | null,
    needsAdapter: boolean,
    adapterType?: string
  } => {
    const sourceOutputs = getComponentOutputs(sourceNodeId);
    const targetInputs = getComponentInputs(targetNodeId);
    
    if (sourceOutputs.length === 0 || targetInputs.length === 0) {
      return { sourceHandle: null, targetHandle: null, needsAdapter: false };
    }
    
    // Pick source handle - prefer primary outputs
    let sourceHandle = sourceOutputs[0]; // Default to first
    if (sourceOutputs.length > 1) {
      // Prefer outputs named in this order: text_output, output, message, data
      const preferredNames = ["text_output", "output", "message", "data"];
      for (const name of preferredNames) {
        const preferred = sourceOutputs.find(o => o.name === name);
        if (preferred) {
          sourceHandle = preferred;
          break;
        }
      }
    }
    
    // Pick target handle - prefer variable handles for Prompt components
    const targetNode = nodes.find(n => n.id === targetNodeId);
    let targetHandle = targetInputs[0]; // Default to first
    
    if (targetNode?.data?.type === "Prompt") {
      // For Prompt components, prefer variable handles
      const promptTemplate = targetNode.data?.node?.template?.template?.value || "";
      const variableMatches = promptTemplate.match(/\{\{([^}]+)\}\}/g);
      
      if (variableMatches && variableMatches.length > 0) {
        const firstVariable = variableMatches[0].replace(/\{\{|\}\}/g, '');
        const variableInput = targetInputs.find(input => input.fieldName === firstVariable);
        if (variableInput) {
          targetHandle = variableInput;
        }
      }
    } else {
      // For other components, prefer common input names by type
      const sourceType = normalizeType(sourceHandle.output_types[0] || "str");
      
      if (sourceType === "str") {
        const preferredNames = ["input", "query", "text", "template", "prompt"];
        for (const name of preferredNames) {
          const preferred = targetInputs.find(input => input.fieldName === name);
          if (preferred) {
            targetHandle = preferred;
            break;
          }
        }
      } else if (sourceType === "Message") {
        const preferredNames = ["message", "chat_history", "input_message"];
        for (const name of preferredNames) {
          const preferred = targetInputs.find(input => input.fieldName === name);
          if (preferred) {
            targetHandle = preferred;
            break;
          }
        }
      }
    }
    
    // Check compatibility
    const sourceType = normalizeType(sourceHandle.output_types[0] || "str");
    const targetType = normalizeType(targetHandle.inputTypes[0] || "str");
    
    if (isCompatible(sourceType, targetType)) {
      return {
        sourceHandle: {
          name: sourceHandle.name,
          output_types: sourceHandle.output_types,
          id: sourceNodeId
        },
        targetHandle: {
          fieldName: targetHandle.fieldName,
          inputTypes: targetHandle.inputTypes,
          id: targetNodeId
        },
        needsAdapter: false
      };
    }
    
    // Types don't match, need adapter
    const adapters = findAdapters(sourceType, targetType);
    if (adapters.length > 0) {
      return {
        sourceHandle: null,
        targetHandle: null,
        needsAdapter: true,
        adapterType: adapters[0] // Use first available adapter
      };
    }
    
    // No adapter available
    return { sourceHandle: null, targetHandle: null, needsAdapter: false };
  };

  // Automatic adapter insertion when types don't match
  const insertAdapter = async (sourceNodeId: string, targetNodeId: string, adapterType: string): Promise<string | null> => {
    try {
      //console.log(`🔧 Inserting adapter ${adapterType} between ${sourceNodeId} and ${targetNodeId}`);
      
      if (!templates[adapterType]) {
        //console.error(`❌ Adapter type ${adapterType} not found in templates`);
        return null;
      }
      
      // Generate a unique ID for the adapter
      const adapterId = `${adapterType}-adapter-${Date.now()}`;
      
      // Add the adapter component to the canvas
      const adapterNodeId = addComponent(templates[adapterType], adapterType, undefined, adapterId);
      //console.log(`✅ Adapter component added with ID: ${adapterNodeId}`);
      
      // Position the adapter between source and target
      const sourceNode = nodes.find(n => n.id === sourceNodeId);
      const targetNode = nodes.find(n => n.id === targetNodeId);
      
      if (sourceNode && targetNode) {
        // Calculate position between source and target
        const sourcePos = sourceNode.position;
        const targetPos = targetNode.position;
        const adapterPos = {
          x: (sourcePos.x + targetPos.x) / 2,
          y: (sourcePos.y + targetPos.y) / 2
        };
        
        // Update adapter position
        useFlowStore.getState().setNode(adapterNodeId, (node) => ({
          ...node,
          position: adapterPos
        }));
        
        //console.log(`🔧 Adapter positioned at:`, adapterPos);
      }
      
      // Create edges: source -> adapter -> target
      // First, create edge from source to adapter
      const sourceToAdapterResult = await createEdgeWithAdapter(sourceNodeId, adapterNodeId);
      if (!sourceToAdapterResult.success) {
        //console.error(`❌ Failed to create edge from source to adapter:`, sourceToAdapterResult.error);
        return null;
      }
      
      // Then, create edge from adapter to target
      const adapterToTargetResult = await createEdgeWithAdapter(adapterNodeId, targetNodeId);
      if (!adapterToTargetResult.success) {
        //console.error(`❌ Failed to create edge from adapter to target:`, adapterToTargetResult.error);
        return null;
      }
      
      //console.log(`✅ Adapter insertion completed successfully`);
      //console.log(`   Source -> Adapter: ${sourceToAdapterResult.success ? '✅' : '❌'}`);
      //console.log(`   Adapter -> Target: ${adapterToTargetResult.success ? '✅' : '❌'}`);
      
      return adapterNodeId;
      
    } catch (error) {
      //console.error(`❌ Adapter insertion failed:`, error);
      return null;
    }
  };

  // Create edge using parsed handle data from component tags
  const createEdgeWithParsedHandles = async (edgeCall: {
    sourceNodeId: string;
    targetNodeId: string;
    sourceHandle: string;
    targetHandle: string;
    sourceHandleData: {
      dataType: string;
      name: string;
      output_types: string[];
    };
    targetHandleData: {
      fieldName: string;
      inputTypes: string[];
      type: string;
    };
  }): Promise<{success: boolean, error?: string}> => {
    try {
      console.log(`🔗 Creating edge with parsed handles:`, edgeCall);
      
      // Get the current nodes and edges from the flow store
      const { setEdges, nodes, edges } = useFlowStore.getState();
      console.log(`🔗 Current nodes in flow:`, nodes.map(n => ({ id: n.id, type: n.data?.type })));
      console.log(`🔗 Looking for source: ${edgeCall.sourceNodeId}, target: ${edgeCall.targetNodeId}`);
      
      // Check if both nodes exist
      const sourceNode = nodes.find(n => n.id === edgeCall.sourceNodeId);
      const targetNode = nodes.find(n => n.id === edgeCall.targetNodeId);
      
      if (!sourceNode) {
        console.log(`❌ Source node ${edgeCall.sourceNodeId} not found in flow`);
        return { success: false, error: `Source node ${edgeCall.sourceNodeId} not found` };
      }
      
      if (!targetNode) {
        console.log(`❌ Target node ${edgeCall.targetNodeId} not found in flow`);
        return { success: false, error: `Target node ${edgeCall.targetNodeId} not found` };
      }
      
      console.log(`✅ Both nodes found:`, { source: sourceNode.id, target: targetNode.id });
      
      // Create the edge with the exact handle data from the parsed tags
      const newEdge = {
        id: `xy-edge__${edgeCall.sourceNodeId}-${edgeCall.targetNodeId}`,
        source: edgeCall.sourceNodeId,
        target: edgeCall.targetNodeId,
        sourceHandle: `{œdataTypeœ:œ${edgeCall.sourceHandleData.dataType}œ,œidœ:œ${edgeCall.sourceNodeId}œ,œnameœ:œ${edgeCall.sourceHandleData.name}œ,œoutput_typesœ:[${edgeCall.sourceHandleData.output_types.map(t => `œ${t}œ`).join(',')}]}`,
        targetHandle: `{œfieldNameœ:œ${edgeCall.targetHandleData.fieldName}œ,œidœ:œ${edgeCall.targetNodeId}œ,œinputTypesœ:[${edgeCall.targetHandleData.inputTypes.map(t => `œ${t}œ`).join(',')}],œtypeœ:œ${edgeCall.targetHandleData.type}œ}`,
        data: {
          sourceHandle: {
            ...edgeCall.sourceHandleData,
            id: edgeCall.sourceNodeId
          },
          targetHandle: {
            ...edgeCall.targetHandleData,
            id: edgeCall.targetNodeId
          }
        }
      };
      
      // Check for duplicate edges
      const existingEdge = edges.find(edge => 
        edge.source === edgeCall.sourceNodeId && 
        edge.target === edgeCall.targetNodeId
      );
      
      if (existingEdge) {
        return { success: false, error: `Edge already exists between ${edgeCall.sourceNodeId} and ${edgeCall.targetNodeId}` };
      }
      
      // Add the edge to the flow store
      setEdges((oldEdges) => [...oldEdges, newEdge]);
      
      //console.log(`✅ Edge created successfully with parsed handles:`, newEdge);
      return { success: true };
      
    } catch (error) {
      //console.error(`❌ Failed to create edge with parsed handles:`, error);
      return { success: false, error: String(error) };
    }
  };

  // Create edge with adapter support
  const createEdgeWithAdapter = async (sourceNodeId: string, targetNodeId: string): Promise<{success: boolean, error?: string}> => {
    try {
      // Use the handle selection algorithm
      const handleSelection = selectHandles(sourceNodeId, targetNodeId);
      
      if (handleSelection.needsAdapter) {
        //console.log(`🔧 Adapter needed for ${sourceNodeId} -> ${targetNodeId}: ${handleSelection.adapterType}`);
        
        if (handleSelection.adapterType) {
          // Insert the adapter
          const adapterId = await insertAdapter(sourceNodeId, targetNodeId, handleSelection.adapterType);
          if (adapterId) {
            // Now try to create edges through the adapter
            const sourceToAdapter = await createEdgeWithAdapter(sourceNodeId, adapterId);
            const adapterToTarget = await createEdgeWithAdapter(adapterId, targetNodeId);
            
            return {
              success: sourceToAdapter.success && adapterToTarget.success,
              error: sourceToAdapter.error || adapterToTarget.error
            };
          } else {
            return { success: false, error: 'Failed to insert adapter' };
          }
        } else {
          return { success: false, error: 'No adapter type specified' };
        }
      }
      
      if (!handleSelection.sourceHandle || !handleSelection.targetHandle) {
        return { success: false, error: 'Could not select compatible handles' };
      }
      
      // Use the validated edge creation
      const result = createValidatedEdge(sourceNodeId, targetNodeId);
      
      if (result.success) {
        //console.log(`✅ Edge created successfully: ${sourceNodeId} -> ${targetNodeId}`);
        if (result.warnings.length > 0) {
          //console.warn(`⚠️ Edge created with warnings:`, result.warnings);
        }
        return { success: true };
      } else {
        //console.error(`❌ Edge creation failed:`, result.errors);
        return { success: false, error: result.errors.join(', ') };
      }
      
    } catch (error) {
      return { success: false, error: String(error) };
    }
  };

  // Edge validation functions
  const validateEdge = (sourceNodeId: string, targetNodeId: string, sourceHandle: any, targetHandle: any): {
    isValid: boolean;
    errors: string[];
    warnings: string[];
  } => {
    const errors: string[] = [];
    const warnings: string[] = [];
    
    // Check if both nodes exist
    const sourceNode = nodes.find(n => n.id === sourceNodeId);
    const targetNode = nodes.find(n => n.id === targetNodeId);
    
    if (!sourceNode) {
      errors.push(`Source node ${sourceNodeId} not found`);
    }
    if (!targetNode) {
      errors.push(`Target node ${targetNodeId} not found`);
    }
    
    if (errors.length > 0) {
      return { isValid: false, errors, warnings };
    }
    
    // Check if source handle exists on source node
    if (sourceNode && sourceHandle) {
      const sourceOutputs = getComponentOutputs(sourceNodeId);
      const sourceHandleExists = sourceOutputs.some(output => output.name === sourceHandle.name);
      if (!sourceHandleExists) {
        errors.push(`Source handle '${sourceHandle.name}' not found on source node ${sourceNodeId}`);
      }
    }
    
    // Check if target handle exists on target node
    if (targetNode && targetHandle) {
      const targetInputs = getComponentInputs(targetNodeId);
      const targetHandleExists = targetInputs.some(input => input.fieldName === targetHandle.fieldName);
      if (!targetHandleExists) {
        errors.push(`Target handle '${targetHandle.fieldName}' not found on target node ${targetNodeId}`);
      }
    }
    
    // Check for circular connections
    if (sourceNodeId === targetNodeId) {
      errors.push(`Cannot connect node to itself: ${sourceNodeId}`);
    }
    
    // Check for duplicate edges
    const existingEdges = useFlowStore.getState().edges;
    const duplicateEdge = existingEdges.find(edge => 
      edge.source === sourceNodeId && edge.target === targetNodeId &&
      edge.sourceHandle === JSON.stringify(sourceHandle) && edge.targetHandle === JSON.stringify(targetHandle)
    );
    
    if (duplicateEdge) {
      warnings.push(`Edge already exists between ${sourceNodeId} and ${targetNodeId} with same handles`);
    }
    
    // Check type compatibility
    if (sourceHandle && targetHandle) {
      const sourceType = normalizeType(sourceHandle.output_types?.[0] || "str");
      const targetType = normalizeType(targetHandle.inputTypes?.[0] || "str");
      
      if (!isCompatible(sourceType, targetType)) {
        warnings.push(`Type mismatch: ${sourceType} -> ${targetType}. Consider using an adapter.`);
      }
    }
    
    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  };

  // Enhanced edge creation with validation
  const createValidatedEdge = (sourceNodeId: string, targetNodeId: string): {
    success: boolean;
    edgeId?: string;
    errors: string[];
    warnings: string[];
  } => {
    const handleSelection = selectHandles(sourceNodeId, targetNodeId);
    
    if (handleSelection.needsAdapter) {
      return {
        success: false,
        errors: [`Adapter needed: ${handleSelection.adapterType}`],
        warnings: []
      };
    }
    
    if (!handleSelection.sourceHandle || !handleSelection.targetHandle) {
      return {
        success: false,
        errors: ['Could not select compatible handles'],
        warnings: []
      };
    }
    
    // Validate the edge before creating it
    const validation = validateEdge(sourceNodeId, targetNodeId, handleSelection.sourceHandle, handleSelection.targetHandle);
    
    if (!validation.isValid) {
      return {
        success: false,
        errors: validation.errors,
        warnings: validation.warnings
      };
    }
    
    // Create the edge
    try {
      createEdgeByIds(sourceNodeId, targetNodeId);
      
      // Generate edge ID for return
      const edgeId = `xy-edge__${sourceNodeId}-${targetNodeId}`;
      
      return {
        success: true,
        edgeId,
        errors: [],
        warnings: validation.warnings
      };
    } catch (error) {
      return {
        success: false,
        errors: [String(error)],
        warnings: validation.warnings
      };
    }
  };

  // Status reporting for edge creation
  const [edgeCreationStatus, setEdgeCreationStatus] = useState<{
    isCreating: boolean;
    currentOperation: string;
    progress: number;
    total: number;
    results: Array<{source: string, target: string, success: boolean, message: string}>;
  }>({
    isCreating: false,
    currentOperation: '',
    progress: 0,
    total: 0,
    results: []
  });

  // State for status display
  const [statusDisplay, setStatusDisplay] = useState({
    sourceNode: false,
    targetNode: false,
    edgeConnected: false
  });

  // Update status display when nodes or edges change
  useEffect(() => {
    const updateStatus = () => {
      const { nodes, edges } = useFlowStore.getState();
      const sourceNode = nodes.find(n => n.id === "TextInput-NEWID");
      const targetNode = nodes.find(n => n.id === "LanguageModelComponent-lRD6W");
      const edgeConnected = edges.some(e => e.source === 'TextInput-NEWID' && e.target === 'LanguageModelComponent-lRD6W');
      
      setStatusDisplay({
        sourceNode: !!sourceNode,
        targetNode: !!targetNode,
        edgeConnected
      });
    };
    
    // Initial update
    updateStatus();
    
    // Subscribe to store changes
    const unsubscribe = useFlowStore.subscribe(updateStatus);
    
    return unsubscribe;
  }, []);

  // Helper function to check if required nodes exist
  const checkRequiredNodes = () => {
    const { nodes } = useFlowStore.getState();
    const sourceNode = nodes.find(n => n.id === "TextInput-NEWID");
    const targetNode = nodes.find(n => n.id === "LanguageModelComponent-lRD6W");
    
    return {
      sourceNode: !!sourceNode,
      targetNode: !!targetNode,
      sourceNodeData: sourceNode,
      targetNodeData: targetNode
    };
  };

  // Function to create the required nodes if they don't exist
  const createRequiredNodes = () => {
    const { setNodes } = useFlowStore.getState();
    const nodeStatus = checkRequiredNodes();
    
    if (nodeStatus.sourceNode && nodeStatus.targetNode) {
      console.log('✅ All required nodes already exist');
      return;
    }
    
    console.log('🔧 Creating missing required nodes...');
    
    if (!nodeStatus.sourceNode) {
      const textInputNode = {
        id: "TextInput-NEWID",
        type: "genericNode",
        position: { x: 100, y: 200 },
        data: {
          type: "TextInput",
          id: "TextInput-NEWID",
          showNode: true,
          node: {
            display_name: "Text Input",
            type: "TextInput",
            description: "Text input component",
            documentation: "",
            tool_mode: false,
            frozen: false,
            template: {
              text: {
                type: "str",
                required: true,
                show: true,
                value: ""
              }
            }
          }
        }
      };
      
      setNodes((oldNodes) => [...oldNodes, textInputNode as any]);
      console.log('✅ Created TextInput-NEWID node');
    }
    
    if (!nodeStatus.targetNode) {
      const llmNode = {
        id: "LanguageModelComponent-lRD6W",
        type: "genericNode",
        position: { x: 400, y: 200 },
        data: {
          type: "LanguageModelComponent",
          id: "LanguageModelComponent-lRD6W",
          showNode: true,
          node: {
            display_name: "Language Model",
            type: "LanguageModelComponent",
            description: "Language model component",
            documentation: "",
            tool_mode: false,
            frozen: false,
            template: {
              system_message: {
                type: "str",
                required: false,
                show: true,
                value: ""
              }
            }
          }
        }
      };
      
      setNodes((oldNodes) => [...oldNodes, llmNode as any]);
      console.log('✅ Created LanguageModelComponent-lRD6W node');
    }
    
    console.log('🎯 All required nodes created successfully');
  };

  // Function to add the specific edge structure provided by the user
  const addSimpleEdge = () => {
    const { setEdges, nodes, edges } = useFlowStore.getState();
    
    // Check if the required nodes exist
    const nodeStatus = checkRequiredNodes();
    
    if (!nodeStatus.sourceNode) {
      console.log('❌ Source node TextInput-NEWID not found in flow');
      console.log('💡 Available nodes:', nodes.map(n => n.id));
      console.log('💡 Use "Create Nodes" button to create missing nodes');
      return;
    }
    
    if (!nodeStatus.targetNode) {
      console.log('❌ Target node LanguageModelComponent-lRD6W not found in flow');
      console.log('💡 Available nodes:', nodes.map(n => n.id));
      console.log('💡 Use "Create Nodes" button to create missing nodes');
      return;
    }
    
    console.log('🔗 Creating specific edge from TextInput-NEWID to LanguageModelComponent-lRD6W');
    console.log('📋 Source node:', nodeStatus.sourceNodeData?.data?.type);
    console.log('📋 Target node:', nodeStatus.targetNodeData?.data?.type);
    
    const newEdge = {
      id: "xy-edge__TextInput-NEWID-LanguageModelComponent-lRD6W",
      source: "TextInput-NEWID",
      target: "LanguageModelComponent-lRD6W",
      sourceHandle: "{œdataTypeœ:œTextInputœ,œidœ:œTextInput-NEWIDœ,œnameœ:œtextœ,œoutput_typesœ:[œMessageœ]}",
      targetHandle: "{œfieldNameœ:œsystem_messageœ,œidœ:œLanguageModelComponent-lRD6Wœ,œinputTypesœ:[œMessageœ],œtypeœ:œstrœ}",
      data: {
        sourceHandle: {
          dataType: "TextInput",
          id: "TextInput-NEWID",
          name: "text",
          output_types: ["Message"]
        },
        targetHandle: {
          fieldName: "system_message",
          id: "LanguageModelComponent-lRD6W",
          inputTypes: ["Message"],
          type: "str"
        }
      }
    };
    
    // Check for duplicate edges
    const existingEdge = edges.find(edge => 
      edge.source === "TextInput-NEWID" && 
      edge.target === "LanguageModelComponent-lRD6W"
    );
    
    if (existingEdge) {
      console.log('⚠️ Edge already exists between TextInput-NEWID and LanguageModelComponent-lRD6W');
      console.log('📋 Existing edge:', existingEdge);
      return;
    }
    
    // Add the edge to the flow store
    setEdges((oldEdges) => [...oldEdges, newEdge]);
    
    console.log('✅ Specific edge created successfully:', newEdge);
    console.log('🎯 Edge ID:', newEdge.id);
    console.log('🔗 Source → Target:', `${newEdge.source} → ${newEdge.target}`);
  };

  // Report edge creation progress
  const reportEdgeProgress = (operation: string, progress: number, total: number) => {
    setEdgeCreationStatus(prev => ({
      ...prev,
      currentOperation: operation,
      progress,
      total
    }));
  };

  // Report edge creation result
  const reportEdgeResult = (source: string, target: string, success: boolean, message: string) => {
    setEdgeCreationStatus(prev => ({
      ...prev,
      results: [...prev.results, { source, target, success, message }]
    }));
  };

  // Clear edge creation status
  const clearEdgeStatus = () => {
    setEdgeCreationStatus({
      isCreating: false,
      currentOperation: '',
      progress: 0,
      total: 0,
      results: []
    });
  };

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
                    //console.log('🔍 Templates object:', templates);
                    //console.log('🔍 Available template keys:', Object.keys(templates));
                    //console.log('🔍 Sample template (first 3):', Object.keys(templates).slice(0, 3).map(key => ({
                    //  key,
                    //  display_name: templates[key]?.display_name,
                    //  description: templates[key]?.description
                    //})));
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
                    const testText = 'Here is a <ChatInput-abc123> component and a <ChatOutput-def456> component';
                    //console.log('🧪 Testing component tag extraction...');
                    //console.log('🧪 Test text:', testText);
                    const extracted = extractComponentTags(testText);
                    //console.log('🧪 Extracted tags:', extracted);
                    
                    // Test with multiple connections
                    const testWithConnections = 'Add a <WebSearchNoAPI-xyz789:ChatInput-abc123,PromptTemplate-def456> component';
                    //console.log('🧪 Test with multiple connections:', testWithConnections);
                    const extractedWithConnections = extractComponentTags(testWithConnections);
                    //console.log('🧪 Extracted with connections:', extractedWithConnections);
                    
                    // Test with actual template keys
                    const firstTemplateKey = Object.keys(templates)[0];
                    if (firstTemplateKey) {
                      const testWithRealComponent = `Add a <${firstTemplateKey}-test123> component`;
                      //console.log('🧪 Test with real component:', testWithRealComponent);
                      const extractedReal = extractComponentTags(testWithRealComponent);
                      //console.log('🧪 Extracted real component:', extractedReal);
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
                     //console.log('🧪 Testing component addition...');
                     const firstTemplateKey = Object.keys(templates)[0];
                     if (firstTemplateKey && flowId) {
                       try {
                         //console.log(`🧪 Adding component: ${firstTemplateKey}`);
                         //console.log(`🧪 Template:`, templates[firstTemplateKey]);
                         addComponent(templates[firstTemplateKey], firstTemplateKey);
                         //console.log(`✅ Test: Added component ${firstTemplateKey}`);
                       } catch (error) {
                         //console.error(`❌ Test: Failed to add component ${firstTemplateKey}:`, error);
                       }
                     } else {
                       //console.warn('🧪 Test: No components or flow available for testing');
                     }
                   }}
                   className="h-6 px-2 text-xs"
                 >
                   <CheckCircle className="h-3 w-3 mr-1" />
                   Test Add
                 </Button>
                 <Button
                   size="sm"
                   variant="outline"
                   onClick={() => {
                     addSimpleEdge();
                   }}
                   className="h-6 px-2 text-xs"
                 >
                   <Zap className="h-3 w-3 mr-1" />
                   Add TextInput→LLM Edge
                 </Button>
                 <Button
                   size="sm"
                   variant="outline"
                   onClick={() => {
                     createRequiredNodes();
                   }}
                   className="h-6 px-2 text-xs"
                 >
                   <Plus className="h-3 w-3 mr-1" />
                   Create Nodes
                 </Button>
                 <Button
                   size="sm"
                   variant="outline"
                   onClick={() => {
                     const status = checkRequiredNodes();
                     console.log('📊 Node Status:', status);
                     console.log('💡 Available nodes:', useFlowStore.getState().nodes.map(n => n.id));
                   }}
                   className="h-6 px-2 text-xs"
                 >
                   <Info className="h-3 w-3 mr-1" />
                   Check Status
                 </Button>
              </div>
              <div className="mb-3 p-2 bg-muted/50 rounded text-xs">
                <div className="font-medium mb-1">Edge Creation Status:</div>
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span>TextInput-NEWID:</span>
                    <span className={`px-2 py-0.5 rounded text-xs ${statusDisplay.sourceNode ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                      {statusDisplay.sourceNode ? '✅ Found' : '❌ Missing'}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span>LanguageModelComponent-lRD6W:</span>
                    <span className={`px-2 py-0.5 rounded text-xs ${statusDisplay.targetNode ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                      {statusDisplay.targetNode ? '✅ Found' : '❌ Missing'}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span>Edge:</span>
                    <span className={`px-2 py-0.5 rounded text-xs ${statusDisplay.edgeConnected ? 'bg-green-100 text-green-800' : 'bg-blue-100 text-blue-800'}`}>
                      {statusDisplay.edgeConnected ? '✅ Connected' : '🔗 Not Connected'}
                    </span>
                  </div>
                </div>
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

