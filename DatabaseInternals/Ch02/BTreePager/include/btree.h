#pragma once
#include "page_manager.h"
#include <optional>

class BTree {
public:
    explicit BTree(PageManager& pager);
    
    void insert(Key key, Value value);
    std::optional<Value> search(Key key);
    
private:
    PageManager& pager_;

    // Recursive insert returning true if split occurred
    bool insert_recursive(PageID node_id, Key key, Value value, Key& new_key, PageID& new_page_id);
    void split_leaf(PageID leaf_id, Page& old_leaf, Key& median_key, PageID& new_leaf_id);
    void split_internal(PageID node_id, Page& old_internal, Key& median_key, PageID& new_internal_id);
    void insert_into_node(Page& node, Key key, Value val, PageID child_id = 0);
};
