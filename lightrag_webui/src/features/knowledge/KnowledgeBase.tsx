import React, { useState } from 'react';
import { ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable';
import { ScrollArea } from '@/components/ui/scroll-area';
import { FileTree } from './components/FileTree';
import { NoteEditor } from './components/NoteEditor';
import { cn } from '@/lib/utils';

export interface Note {
  id: string;
  title: string;
  content: string;
  path: string;
}

export const KnowledgeBase: React.FC = () => {
  const [selectedNote, setSelectedNote] = useState<Note | null>(null);

  return (
    <div className="h-screen w-full">
      <ResizablePanelGroup direction="horizontal">
        <ResizablePanel defaultSize={20} minSize={15} maxSize={40}>
          <div className="h-full border-r">
            <ScrollArea className="h-full">
              <div className="p-4">
                <FileTree onSelectNote={setSelectedNote} />
              </div>
            </ScrollArea>
          </div>
        </ResizablePanel>
        <ResizablePanel defaultSize={80}>
          <div className={cn("h-full", !selectedNote && "flex items-center justify-center text-muted-foreground")}>
            {selectedNote ? (
              <NoteEditor note={selectedNote} />
            ) : (
              <div>选择或创建一个笔记开始编辑</div>
            )}
          </div>
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
}; 