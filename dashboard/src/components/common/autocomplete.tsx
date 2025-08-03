"use client";

import {
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandLoading,
} from "@/components/ui/command";
import { cn } from "@/lib/utils";
import { Command as CommandPrimitive } from "cmdk";
import { Check } from "lucide-react";
import {
  type KeyboardEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

export type Option = Record<"value" | "label", string> & Record<string, string>;

type AutoCompleteProps = {
  options: Option[];
  emptyMessage: string;
  value?: Option;
  onValueChange?: (value: Option) => void;
  isLoading?: boolean;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
  allowCustomValue?: boolean;
};

export const AutoComplete = ({
  options,
  placeholder,
  emptyMessage,
  value,
  onValueChange,
  disabled,
  isLoading = false,
  className = "",
  allowCustomValue = true,
}: AutoCompleteProps) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isOpen, setOpen] = useState(false);
  const [selected, setSelected] = useState<Option>(value as Option);
  const [inputValue, setInputValue] = useState<string>(value?.label || "");

  // Sync internal state when value prop changes
  useEffect(() => {
    setSelected(value as Option);
    setInputValue(value?.label || "");
  }, [value]);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      const input = inputRef.current;
      if (!input) return;

      if (!isOpen) {
        setOpen(true);
      }

      if (event.key === "Enter" && input.value !== "") {
        // Try selecting existing option
        const match = options.find((opt) => opt.label === input.value);
        if (match) {
          setSelected(match);
          onValueChange?.(match);
        } else if (allowCustomValue) {
          // Create and select custom value
          const customOption = { label: input.value, value: input.value };
          setSelected(customOption);
          onValueChange?.(customOption);
        }
      }

      if (event.key === "Escape") {
        input.blur();
      }
    },
    [isOpen, options, onValueChange, allowCustomValue],
  );

  const handleBlur = useCallback(() => {
    setOpen(false);
    setInputValue(selected?.label || "");
  }, [selected]);

  const handleSelectOption = useCallback(
    (option: Option) => {
      setInputValue(option.label || "");
      setSelected(option);
      onValueChange?.(option);
      // blur to close
      setTimeout(() => inputRef.current?.blur(), 0);
    },
    [onValueChange],
  );

  const hasCustom =
    allowCustomValue &&
    inputValue &&
    !options.some((opt) => opt.label === inputValue);

  return (
    <CommandPrimitive onKeyDown={handleKeyDown} className={className}>
      <div>
        <CommandInput
          ref={inputRef}
          value={inputValue}
          onValueChange={setInputValue} // Always allow typing, even when loading
          onBlur={handleBlur}
          onFocus={() => setOpen(true)}
          placeholder={placeholder}
          disabled={disabled}
          className="text-base"
        />
      </div>
      <div className="relative mt-1">
        <div
          className={cn(
            "animate-in fade-in-0 zoom-in-95 absolute top-0 z-10 w-full rounded-xl outline-none",
            isOpen ? "block" : "hidden",
          )}
        >
          <CommandList className="rounded-lg ring-1 bg-card">
            {isLoading && <CommandLoading isLoading={isLoading} />}

            {options.length > 0 && (
              <CommandGroup>
                {options.map((option) => {
                  const isSelected = selected?.value === option.value;
                  return (
                    <CommandItem
                      key={option.value}
                      value={option.label}
                      onMouseDown={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                      }}
                      onSelect={() => handleSelectOption(option)}
                      className={cn(
                        "flex w-full items-center gap-2",
                        !isSelected && "pl-8",
                      )}
                    >
                      {isSelected && <Check className="w-4" />}
                      {option.label}
                    </CommandItem>
                  );
                })}
              </CommandGroup>
            )}

            {hasCustom && (
              <CommandGroup>
                <CommandItem
                  value={inputValue}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                  }}
                  onSelect={() =>
                    handleSelectOption({ label: inputValue, value: inputValue })
                  }
                  className="pl-8"
                >
                  Use "{inputValue}"
                </CommandItem>
              </CommandGroup>
            )}

            {!isLoading && options.length === 0 && !hasCustom && (
              <CommandPrimitive.Empty className="select-none rounded-sm px-2 py-3 text-center text-sm">
                {emptyMessage}
              </CommandPrimitive.Empty>
            )}
          </CommandList>
        </div>
      </div>
    </CommandPrimitive>
  );
};
