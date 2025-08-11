import type { UseMutationResult } from "@tanstack/react-query";
import type { useMutationFunctionType } from "@/types/api";
import { api } from "../../api";
import { getURL } from "../../helpers/constants";
import { UseRequestProcessor } from "../../services/request-processor";

type AddNodeOp = { op: "add_node"; node: Record<string, any> };
type UpdateNodeOp = { op: "update_node"; id: string; set: Record<string, any> };
type RemoveNodeOp = { op: "remove_node"; id: string };
type AddEdgeOp = { op: "add_edge"; edge: Record<string, any> };
type RemoveEdgeOp = { op: "remove_edge"; id: string };

export type FlowOperation =
  | AddNodeOp
  | UpdateNodeOp
  | RemoveNodeOp
  | AddEdgeOp
  | RemoveEdgeOp;

interface IApplyFlowOpsRequest {
  id: string;
  base_version?: string | null; // ISO datetime from flow.updated_at
  operations: FlowOperation[];
}

export const usePostApplyFlowOps: useMutationFunctionType<undefined, IApplyFlowOpsRequest> = (
  options?,
) => {
  const { mutate, queryClient } = UseRequestProcessor();

  const ApplyFlowOpsFn = async ({ id, base_version, operations }: IApplyFlowOpsRequest) => {
    const payload = { base_version, operations };
    
    // Console log the API call being made in copy-pasteable format
    console.log("=== APPLYING FLOW OPERATIONS - API CALL DETAILS ===");
    console.log("URL:", `${getURL("FLOWS")}/${id}/ops`);
    console.log("Method: POST");
    console.log("Flow ID:", id);
    console.log("Base Version:", base_version);
    console.log("Request Body:");
    console.log(JSON.stringify(payload, null, 2));
    console.log("=== END API CALL DETAILS ===");
    
    const response = await api.post(`${getURL("FLOWS")}/${id}/ops`, payload);
    return response.data;
  };

  const mutation: UseMutationResult<IApplyFlowOpsRequest, any, IApplyFlowOpsRequest> = mutate(
    ["usePostApplyFlowOps"],
    ApplyFlowOpsFn,
    {
      onSettled: (res) => {
        // refresh the flow details and folders
        queryClient.refetchQueries({ queryKey: ["useGetFlow", res?.id] });
        queryClient.refetchQueries({ queryKey: ["useGetFolders", (res as any)?.folder_id] });
        queryClient.refetchQueries({ queryKey: ["useGetFolder"] });
      },
      ...options,
    },
  );

  return mutation;
};


