import { useStoreApi } from "@xyflow/react";
import { useCallback } from "react";
import { NODE_WIDTH } from "@/constants/constants";
import { track } from "@/customization/utils/analytics";
import useFlowStore from "@/stores/flowStore";
import type { APIClassType } from "@/types/api";
import type { AllNodeType } from "@/types/flow";
import { getNodeId } from "@/utils/reactflowUtils";
import { getNodeRenderType } from "@/utils/utils";

export function useAddComponent() {
  const store = useStoreApi();
  const paste = useFlowStore((state) => state.paste);

  const addComponent = useCallback(
    (
      component: APIClassType,
      type: string,
      position?: { x: number; y: number },
      customId?: string,
    ): string => {
      track("Component Added", { componentType: component.display_name });

      const {
        height,
        width,
        transform: [transformX, transformY, zoomLevel],
      } = store.getState();

      const zoomMultiplier = 1 / zoomLevel;

      let pos;

      if (position) {
        pos = position;
      } else {
        let centerX, centerY;

        centerX = -transformX * zoomMultiplier + (width * zoomMultiplier) / 2;
        centerY = -transformY * zoomMultiplier + (height * zoomMultiplier) / 2;

        const nodeOffset = NODE_WIDTH / 2;

        pos = {
          x: -nodeOffset,
          y: -nodeOffset,
          paneX: centerX,
          paneY: centerY,
        };
      }

      const newId = customId || getNodeId(type);

      const newNode: AllNodeType = {
        id: newId,
        type: getNodeRenderType("genericnode"),
        position: { x: 0, y: 0 },
        data: {
          node: component,
          showNode: !component.minimized,
          type: type,
          id: newId,
        },
      };

      if (customId) {
        // If custom ID is provided, add directly to flow store to preserve the ID
        const { setNodes } = useFlowStore.getState();
        setNodes((oldNodes) => [...oldNodes, newNode]);
      } else {
        // Use paste for default behavior when no custom ID
        paste({ nodes: [newNode], edges: [] }, pos);
      }
      
      // Return the generated node ID so callers can use it for edge creation
      return newId;
    },
    [store, paste],
  );

  return addComponent;
}
