import React, { useState, useEffect } from 'react';
import MDEditor from '@uiw/react-md-editor';
import { Note } from '../KnowledgeBase';

interface NoteEditorProps {
  note: Note;
}

export const NoteEditor: React.FC<NoteEditorProps> = ({ note }) => {
  const [content, setContent] = useState(note.content);

  useEffect(() => {
    setContent(note.content);
  }, [note.id]);

  const handleChange = (value?: string) => {
    if (value !== undefined) {
      setContent(value);
      // TODO: 添加自动保存功能
    }
  };

  return (
    <div className="h-full flex flex-col p-4">
      <div className="mb-4">
        <h1 className="text-2xl font-bold">{note.title}</h1>
      </div>
      <div className="flex-1 overflow-hidden">
        <MDEditor
          value={content}
          onChange={handleChange}
          height="100%"
          preview="edit"
          hideToolbar={false}
          enableScroll={true}
        />
      </div>
    </div>
  );
}; 