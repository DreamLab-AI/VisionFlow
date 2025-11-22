/**
 * Notification container for displaying toast messages
 */

import { useUIStore } from '../../stores/useUIStore';
import { ToastContainer } from "@/components/ui/toast"

export function NotificationContainer() {
  const { notifications, removeNotification } = useUIStore();

  return (
    <ToastContainer
      toasts={notifications}
      onClose={removeNotification}
    />
  );
}
