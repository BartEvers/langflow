import { useState } from "react";
import ForwardedIconComponent from "@/components/common/genericIconComponent";
import ShadTooltip from "@/components/common/shadTooltipComponent";
import AireliusModal from "@/components/airelius/AireliusModal";

interface AireliusButtonProps {
  canvasOpen: boolean;
}

const AireliusIcon = () => (
  <ForwardedIconComponent
    name="Wand2"
    className="h-4 w-4 transition-all"
    strokeWidth={2}
  />
);

const ButtonLabel = () => (
  <span className="hidden md:block">Airelius</span>
);

const AireliusButton = ({ canvasOpen }: AireliusButtonProps) => {
  const [open, setOpen] = useState<boolean>(false);

  return (
    <>
      <ShadTooltip content="AI-powered flow generation and planning">
        <div
          data-testid="airelius-btn-flow-toolbar"
          className="airelius-btn-flow-toolbar hover:bg-accent cursor-pointer"
          onClick={() => setOpen(true)}
        >
          <AireliusIcon />
          <ButtonLabel />
        </div>
      </ShadTooltip>
      
      <AireliusModal
        open={open}
        setOpen={setOpen}
        canvasOpen={canvasOpen}
      />
    </>
  );
};

export default AireliusButton;
