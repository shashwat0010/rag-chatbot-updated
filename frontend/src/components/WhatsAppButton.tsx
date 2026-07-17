"use client";

import { useEffect, useState } from "react";

export function WhatsAppButton() {
  const [shareUrl, setShareUrl] = useState("");

  useEffect(() => {
    // Safely capture window.location.href once mounted on the client
    setShareUrl(window.location.href);
  }, []);

  const phoneNumber = process.env.NEXT_PUBLIC_WHATSAPP_NUMBER || "";
  const customMessage = process.env.NEXT_PUBLIC_WHATSAPP_MESSAGE || "Check out this Medical Research Assistant!";

  // Generate WhatsApp API URL
  // If a number is specified, it opens a direct chat with that number.
  // Otherwise, it acts as a share button sharing the website link.
  const whatsappUrl = phoneNumber
    ? `https://wa.me/${phoneNumber.replace(/[^0-9]/g, "")}?text=${encodeURIComponent(customMessage)}`
    : `https://wa.me/?text=${encodeURIComponent(customMessage + " " + shareUrl)}`;

  return (
    <div className="fixed top-[88px] right-4 z-40 sm:top-auto sm:bottom-6 sm:right-6">
      <a
        href={whatsappUrl}
        target="_blank"
        rel="noopener noreferrer"
        className="relative flex h-11 w-11 sm:h-14 sm:w-14 items-center justify-center rounded-full bg-gradient-to-br from-[#25D366] to-[#128C7E] text-white shadow-xl transition-all duration-300 hover:scale-110 focus:outline-none focus:ring-2 focus:ring-[#25D366] focus:ring-offset-2 dark:focus:ring-offset-slate-900"
        aria-label={phoneNumber ? "Chat with us on WhatsApp" : "Share on WhatsApp"}
        title={phoneNumber ? "Chat with us on WhatsApp" : "Share on WhatsApp"}
      >
        {/* Pulsing Outer Ring */}
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[#25D366] opacity-30"></span>
        
        {/* WhatsApp SVG Icon */}
        <svg
          viewBox="0 0 24 24"
          className="relative z-10 h-5 w-5 sm:h-7 sm:w-7 fill-current"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path d="M12.012 2c-5.506 0-9.989 4.478-9.99 9.984a9.96 9.96 0 0 0 1.333 4.993L2 22l5.13-1.347a9.947 9.947 0 0 0 4.88 1.277h.005c5.505 0 9.989-4.478 9.99-9.985A9.98 9.98 0 0 0 12.012 2zm5.835 14.165c-.244.688-1.22 1.259-1.688 1.3a3.528 3.528 0 0 1-1.748-.258c-.732-.303-1.646-.826-2.584-1.643-1.353-1.18-2.227-2.617-2.617-3.267-.39-.65-.7-1.182-.7-1.716 0-1.11.536-1.57.78-1.84.186-.206.41-.31.614-.31a.885.885 0 0 1 .613.23c.204.205.513.784.558.875.067.135.112.293.023.473-.09.18-.18.315-.27.428-.09.18-.18.315-.27.428-.09.112-.192.203-.27.315-.09.112-.18.225-.078.405.102.18.452.743.966 1.204.664.595 1.222.778 1.393.868.17.09.27.067.363-.045.093-.112.408-.473.52-.63.114-.157.227-.135.385-.068.158.068 1.002.473 1.172.563.17.09.283.135.328.214.045.078.045.45-.199 1.138z" />
        </svg>
      </a>
    </div>
  );
}

