import { useState, useRef, useEffect } from 'react'

function SearchableSelect({ options, value, onChange, placeholder, icon }) {
  const [isOpen, setIsOpen] = useState(false)
  const [search, setSearch] = useState('')
  const wrapperRef = useRef(null)
  const inputRef = useRef(null)

  // Get selected option label
  const selectedOption = options.find(opt => opt.id === value)
  
  // Filter options based on search
  const filteredOptions = options.filter(opt => 
    opt.name.toLowerCase().includes(search.toLowerCase())
  )

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event) {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target)) {
        setIsOpen(false)
        setSearch('')
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  // Focus input when dropdown opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus()
    }
  }, [isOpen])

  const handleSelect = (optionId) => {
    onChange(optionId)
    setIsOpen(false)
    setSearch('')
  }

  return (
    <div className="searchable-select" ref={wrapperRef}>
      <div 
        className={`select-trigger ${isOpen ? 'open' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
      >
        {icon && <i className={icon}></i>}
        <span className={selectedOption ? 'selected-text' : 'placeholder-text'}>
          {selectedOption ? selectedOption.name : placeholder}
        </span>
        <i className={`fa-solid fa-chevron-down chevron ${isOpen ? 'rotated' : ''}`}></i>
      </div>
      
      {isOpen && (
        <div className="select-dropdown">
          <div className="search-wrapper">
            <i className="fa-solid fa-magnifying-glass"></i>
            <input
              ref={inputRef}
              type="text"
              placeholder="Type to search..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              onClick={(e) => e.stopPropagation()}
            />
          </div>
          
          <div className="options-list">
            {filteredOptions.length > 0 ? (
              filteredOptions.map(opt => (
                <div
                  key={opt.id}
                  className={`option-item ${opt.id === value ? 'selected' : ''}`}
                  onClick={() => handleSelect(opt.id)}
                >
                  {opt.name}
                  {opt.abbr && <span className="option-abbr">{opt.abbr}</span>}
                </div>
              ))
            ) : (
              <div className="no-results">No results found</div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default SearchableSelect
