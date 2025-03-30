import React, { useState } from 'react';
import { ChevronDown, ChevronRight, File, Folder, Plus, Trash2 } from 'lucide-react';
import Button from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from '@/components/ui/context-menu';
import { Note } from '../KnowledgeBase';

interface FileTreeProps {
  onSelectNote: (note: Note) => void;
}

interface TreeItem {
  id: string;
  name: string;
  type: 'folder' | 'file';
  children?: TreeItem[];
}

export const FileTree: React.FC<FileTreeProps> = ({ onSelectNote }) => {
  const [items, setItems] = useState<TreeItem[]>([]);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());
  const [isCreating, setIsCreating] = useState(false);
  const [newItemName, setNewItemName] = useState('');
  const [newItemParentId, setNewItemParentId] = useState<string | null>(null);
  const [newItemType, setNewItemType] = useState<'folder' | 'file'>('file');

  const toggleFolder = (folderId: string) => {
    setExpandedFolders(prev => {
      const next = new Set(prev);
      if (next.has(folderId)) {
        next.delete(folderId);
      } else {
        next.add(folderId);
      }
      return next;
    });
  };

  const handleCreateItem = (parentId: string | null, type: 'folder' | 'file') => {
    setIsCreating(true);
    setNewItemParentId(parentId);
    setNewItemType(type);
    setNewItemName('');
  };

  const handleCreateSubmit = () => {
    if (!newItemName.trim()) return;

    const newItem: TreeItem = {
      id: Math.random().toString(36).substr(2, 9),
      name: newItemName,
      type: newItemType,
      children: newItemType === 'folder' ? [] : undefined,
    };

    setItems(prev => {
      if (!newItemParentId) {
        return [...prev, newItem];
      }

      const updateChildren = (items: TreeItem[]): TreeItem[] => {
        return items.map(item => {
          if (item.id === newItemParentId) {
            return {
              ...item,
              children: [...(item.children || []), newItem],
            };
          }
          if (item.children) {
            return {
              ...item,
              children: updateChildren(item.children),
            };
          }
          return item;
        });
      };

      return updateChildren(prev);
    });

    setIsCreating(false);
  };

  const renderItem = (item: TreeItem, level: number = 0) => {
    const isExpanded = expandedFolders.has(item.id);
    const isFolder = item.type === 'folder';

    return (
      <div key={item.id}>
        <ContextMenu>
          <ContextMenuTrigger>
            <div
              className="flex items-center py-1 px-2 hover:bg-accent rounded-sm cursor-pointer"
              style={{ paddingLeft: `${level * 12}px` }}
              onClick={() => isFolder ? toggleFolder(item.id) : onSelectNote({
                id: item.id,
                title: item.name,
                content: '',
                path: item.name,
              })}
            >
              <Button variant="ghost" size="icon" className="h-4 w-4">
                {isFolder ? (
                  isExpanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />
                ) : null}
              </Button>
              {isFolder ? (
                <Folder className="h-4 w-4 mr-2" />
              ) : (
                <File className="h-4 w-4 mr-2" />
              )}
              <span className="text-sm">{item.name}</span>
            </div>
          </ContextMenuTrigger>
          <ContextMenuContent>
            {isFolder && (
              <>
                <ContextMenuItem onClick={() => handleCreateItem(item.id, 'folder')}>
                  新建文件夹
                </ContextMenuItem>
                <ContextMenuItem onClick={() => handleCreateItem(item.id, 'file')}>
                  新建笔记
                </ContextMenuItem>
              </>
            )}
            <ContextMenuItem className="text-destructive">
              删除
            </ContextMenuItem>
          </ContextMenuContent>
        </ContextMenu>
        {isFolder && isExpanded && item.children?.map(child => renderItem(child, level + 1))}
      </div>
    );
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">笔记</h2>
        <div className="space-x-1">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => handleCreateItem(null, 'folder')}
          >
            <Plus className="h-4 w-4" />
          </Button>
        </div>
      </div>
      {isCreating && (
        <div className="flex items-center space-x-2">
          <Input
            size={1}
            value={newItemName}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewItemName(e.target.value)}
            onKeyDown={(e: React.KeyboardEvent) => e.key === 'Enter' && handleCreateSubmit()}
            placeholder={newItemType === 'folder' ? '新文件夹名称' : '新笔记名称'}
            autoFocus
          />
        </div>
      )}
      <div className="space-y-1">
        {items.map(item => renderItem(item))}
      </div>
    </div>
  );
}; 