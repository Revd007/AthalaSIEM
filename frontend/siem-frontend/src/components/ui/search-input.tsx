import * as React from 'react';

interface SearchInputProps {
    placeholder?: string;
    onSearch: (value: string) => void;
  }
  
  export function SearchInput(props: SearchInputProps) {
    const { placeholder, onSearch } = props;
    const [value, setValue] = React.useState('');

    const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
      const newValue = event.target.value;
      setValue(newValue);
      onSearch(newValue);
    };

    return (
      <div className="relative">
        <input
          type="text"
          className="w-full h-10 px-3 py-2 text-sm border rounded-md border-input bg-background ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
          placeholder={placeholder}
          value={value}
          onChange={handleChange}
        />
      </div>
    );
  }