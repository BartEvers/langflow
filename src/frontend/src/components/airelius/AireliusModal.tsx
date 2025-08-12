import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import ForwardedIconComponent from "@/components/common/genericIconComponent";
import PFUPlanning from "./PFUPlanning";
import PFUExecution from "./PFUExecution";
import FileIndexing from "./FileIndexing";

interface AireliusModalProps {
  open: boolean;
  setOpen: (open: boolean) => void;
  canvasOpen: boolean;
}

const AireliusModal = ({ open, setOpen, canvasOpen }: AireliusModalProps) => {
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-hidden">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <ForwardedIconComponent name="Wand2" className="h-5 w-5" />
            Airelius - AI Flow Generation
          </DialogTitle>
        </DialogHeader>
        
        <Tabs defaultValue="planning" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="planning">Planning</TabsTrigger>
            <TabsTrigger value="execution">Execution</TabsTrigger>
            <TabsTrigger value="indexing">File Indexing</TabsTrigger>
          </TabsList>
          
          <TabsContent value="planning" className="mt-4">
            <PFUPlanning />
          </TabsContent>
          
          <TabsContent value="execution" className="mt-4">
            <PFUExecution />
          </TabsContent>
          
          <TabsContent value="indexing" className="mt-4">
            <FileIndexing />
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
};

export default AireliusModal;
