import * as React from "react"
import { cn } from "@/lib/utils"
import { X, CheckCircle, AlertCircle, AlertTriangle, Info } from "lucide-react"

export interface ToastProps {
  id: string
  message: string
  type: "success" | "error" | "warning" | "info"
  onClose: (id: string) => void
}

const typeStyles = {
  success: "border-green-500 bg-card",
  error: "border-red-500 bg-card",
  warning: "border-amber-500 bg-card",
  info: "border-blue-500 bg-card"
}

const iconMap = {
  success: <CheckCircle className="h-5 w-5 text-green-500" />,
  error: <AlertCircle className="h-5 w-5 text-red-500" />,
  warning: <AlertTriangle className="h-5 w-5 text-amber-500" />,
  info: <Info className="h-5 w-5 text-blue-500" />
}

export function Toast({ id, message, type, onClose }: ToastProps) {
  return (
    <div
      className={cn(
        "flex items-start gap-3 p-4 rounded-lg border-l-4 shadow-lg animate-in slide-in-from-right-full pointer-events-auto",
        typeStyles[type]
      )}
      role="alert"
    >
      <div className="flex-shrink-0 mt-0.5">
        {iconMap[type]}
      </div>

      <div className="flex-1 min-w-0">
        <p className="text-sm text-gray-900 dark:text-gray-100 break-words">
          {message}
        </p>
      </div>

      <button
        onClick={() => onClose(id)}
        className="flex-shrink-0 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
        aria-label="Close notification"
      >
        <X className="h-5 w-5" />
      </button>
    </div>
  )
}

export interface ToastContainerProps {
  toasts: ToastProps[]
  onClose: (id: string) => void
}

export function ToastContainer({ toasts, onClose }: ToastContainerProps) {
  if (toasts.length === 0) {
    return null
  }

  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-3 max-w-md pointer-events-none">
      {toasts.map((toast) => (
        <Toast key={toast.id} {...toast} onClose={onClose} />
      ))}
    </div>
  )
}
