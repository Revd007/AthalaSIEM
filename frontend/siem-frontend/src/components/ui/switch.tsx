import * as React from "react"
import * as SwitchPrimitives from "@radix-ui/react-switch"

const Switch = React.forwardRef<
  React.ElementRef<typeof SwitchPrimitives.Root>,
  React.ComponentPropsWithoutRef<typeof SwitchPrimitives.Root>
>(({ className, ...props }, ref) => (
  <SwitchPrimitives.Root
    className="w-10 h-6 bg-gray-200 rounded-full transition-colors data-[state=checked]:bg-indigo-600"
    {...props}
    ref={ref}
  >
    <SwitchPrimitives.Thumb className="block w-5 h-5 bg-white rounded-full transition-transform data-[state=checked]:translate-x-4" />
  </SwitchPrimitives.Root>
))
Switch.displayName = "Switch"

export { Switch }