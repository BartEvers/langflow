import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2, Play, CheckCircle, XCircle } from "lucide-react";
import useFlowStore from "@/stores/flowStore";

const PFUExecution = () => {
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionResult, setExecutionResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  
  const currentFlow = useFlowStore((state) => state.currentFlow);
  const flowId = useFlowStore((state) => state.currentFlow?.id);

  const executePlan = async (plan: any) => {
    if (!plan || !flowId) return;
    
    setIsExecuting(true);
    setError(null);
    
    try {
      // Console log the API call being made in copy-pasteable format
      console.log("=== EXECUTING PFU PLAN (PFUExecution) - API CALL DETAILS ===");
      console.log("URL: /api/v1/airelius/pfu/execute");
      console.log("Method: POST");
      console.log("Flow ID:", flowId);
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
      
      const data = await response.json();
      setExecutionResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to execute plan");
    } finally {
      setIsExecuting(false);
    }
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Play className="h-5 w-5" />
            Execute PFU Plan
          </CardTitle>
          <CardDescription>
            Execute the generated plan step by step with validation
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground mb-4">
            First generate a plan in the Planning tab, then execute it here.
          </p>
          
          {executionResult && (
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-sm">
                <CheckCircle className="h-4 w-4 text-green-600" />
                <span>Execution completed successfully!</span>
              </div>
              
              <div className="bg-muted p-3 rounded-md">
                <h4 className="font-semibold text-sm mb-2">Execution Summary:</h4>
                <div className="text-xs space-y-1">
                  <div>Total Steps: {executionResult.execution_summary?.total_steps}</div>
                  <div>Successful: {executionResult.execution_summary?.successful_steps}</div>
                  <div>Failed: {executionResult.execution_summary?.failed_steps}</div>
                </div>
              </div>
            </div>
          )}
          
          {error && (
            <div className="text-sm text-red-600 bg-red-50 p-3 rounded-md flex items-center gap-2">
              <XCircle className="h-4 w-4" />
              {error}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default PFUExecution;