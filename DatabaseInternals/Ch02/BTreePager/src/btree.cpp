#include "btree.h"
#include <algorithm>
#include <iostream>
#include <cstring>

BTree::BTree(PageManager& pager) : pager_(pager) {}

void BTree::insert(Key key, Value value) {
    PageID root = pager_.root_id();
    Key new_key = 0;
    PageID new_page = 0;

    if (insert_recursive(root, key, value, new_key, new_page)) {
        // Root split: create new root pointing to old root + new sibling
        PageID new_root = pager_.allocate_page();
        
        Page new_root_page = {};
        std::memset(&new_root_page, 0, sizeof(new_root_page));
        new_root_page.type = NodeType::INTERNAL;
        new_root_page.count = 1;
        new_root_page.parent_id = 0;
        new_root_page.keys[0] = new_key;
        new_root_page.child_ids[0] = root;
        new_root_page.child_ids[1] = new_page;
        
        pager_.write_page(new_root, new_root_page);
        pager_.set_root(new_root);
    }
}

std::optional<Value> BTree::search(Key key) {
    PageID current = pager_.root_id();
    while (true) {
        Page node;
        pager_.read_page(current, node);
        
        if (node.type == NodeType::LEAF) {
            // Linear scan within leaf (could binary search for large pages)
            for (int i = 0; i < node.count; ++i) {
                if (node.keys[i] == key) {
                    return node.values[i];
                }
            }
            return std::nullopt;
        }
        
        // Internal node: find child to descend into
        int i = 0;
        while (i < node.count && key >= node.keys[i]) ++i;
        current = node.child_ids[i];
    }
}

bool BTree::insert_recursive(PageID node_id, Key key, Value value, Key& new_key, PageID& new_page_id) {
    Page node;
    pager_.read_page(node_id, node);
    
    if (node.type == NodeType::LEAF) {
        // Check for duplicate key — update in place
        for (int i = 0; i < node.count; ++i) {
            if (node.keys[i] == key) {
                node.values[i] = value;
                pager_.write_page(node_id, node);
                return false;
            }
        }
        
        if (node.count >= MAX_ENTRIES_PER_PAGE) {
            // Insert first, then split (ensures key is placed)
            // We need to split: insert into a temporary overflow, then split
            split_leaf(node_id, node, new_key, new_page_id);
            
            // Now insert into the correct half
            if (key < new_key) {
                // Re-read the old (left) leaf after split
                pager_.read_page(node_id, node);
                insert_into_node(node, key, value);
                pager_.write_page(node_id, node);
            } else {
                // Insert into new (right) leaf
                Page new_leaf;
                pager_.read_page(new_page_id, new_leaf);
                insert_into_node(new_leaf, key, value);
                pager_.write_page(new_page_id, new_leaf);
            }
            return true;
        }
        
        insert_into_node(node, key, value);
        pager_.write_page(node_id, node);
        return false;
    } else {
        // Internal node: find child to descend into
        int i = 0;
        while (i < node.count && key >= node.keys[i]) ++i;
        PageID child = node.child_ids[i];
        
        Key child_new_key = 0;
        PageID child_new_page = 0;
        bool split = insert_recursive(child, key, value, child_new_key, child_new_page);
        
        if (split) {
            if (node.count >= MAX_ENTRIES_PER_PAGE) {
                // Need to split this internal node too
                split_internal(node_id, node, new_key, new_page_id);
                
                // Insert the propagated key into the correct half
                if (child_new_key < new_key) {
                    pager_.read_page(node_id, node);
                    insert_into_node(node, child_new_key, 0, child_new_page);
                    pager_.write_page(node_id, node);
                } else {
                    Page new_internal;
                    pager_.read_page(new_page_id, new_internal);
                    insert_into_node(new_internal, child_new_key, 0, child_new_page);
                    pager_.write_page(new_page_id, new_internal);
                }
                return true;
            }
            
            // Re-read node (may have been modified by recursive calls)
            pager_.read_page(node_id, node);
            insert_into_node(node, child_new_key, 0, child_new_page);
            pager_.write_page(node_id, node);
            return false;
        }
        return false;
    }
}

void BTree::split_leaf(PageID leaf_id, Page& old_leaf, Key& median_key, PageID& new_leaf_id) {
    new_leaf_id = pager_.allocate_page();
    Page new_leaf = {};
    std::memset(&new_leaf, 0, sizeof(new_leaf));
    new_leaf.type = NodeType::LEAF;
    new_leaf.parent_id = old_leaf.parent_id;
    
    int mid = old_leaf.count / 2;
    median_key = old_leaf.keys[mid];
    
    // Move second half to new leaf
    int new_count = 0;
    for (int i = mid; i < old_leaf.count; ++i) {
        new_leaf.keys[new_count] = old_leaf.keys[i];
        new_leaf.values[new_count] = old_leaf.values[i];
        new_count++;
    }
    new_leaf.count = static_cast<uint16_t>(new_count);
    
    // Clear moved entries from old leaf
    for (int i = mid; i < old_leaf.count; ++i) {
        old_leaf.keys[i] = 0;
        old_leaf.values[i] = 0;
    }
    old_leaf.count = static_cast<uint16_t>(mid);
    
    // Link siblings for future range scans
    new_leaf.next_sibling_id = old_leaf.next_sibling_id;
    old_leaf.next_sibling_id = new_leaf_id;
    
    pager_.write_page(new_leaf_id, new_leaf);
    pager_.write_page(leaf_id, old_leaf);
}

void BTree::split_internal(PageID node_id, Page& old_internal, Key& median_key, PageID& new_internal_id) {
    new_internal_id = pager_.allocate_page();
    Page new_internal = {};
    std::memset(&new_internal, 0, sizeof(new_internal));
    new_internal.type = NodeType::INTERNAL;
    new_internal.parent_id = old_internal.parent_id;
    
    int mid = old_internal.count / 2;
    median_key = old_internal.keys[mid];
    
    // Move keys and children after the median to the new node
    // The median key itself is promoted upward (not stored in either child)
    int new_count = 0;
    for (int i = mid + 1; i < old_internal.count; ++i) {
        new_internal.keys[new_count] = old_internal.keys[i];
        new_internal.child_ids[new_count] = old_internal.child_ids[i];
        new_count++;
    }
    // Last child pointer
    new_internal.child_ids[new_count] = old_internal.child_ids[old_internal.count];
    new_internal.count = static_cast<uint16_t>(new_count);
    
    // Clear moved entries from old node
    // NOTE: keys[] has MAX_ENTRIES_PER_PAGE elements (indices 0..199)
    //       child_ids[] has MAX_ENTRIES_PER_PAGE+1 elements (indices 0..200)
    //       Must NOT access keys[count] when count==MAX_ENTRIES_PER_PAGE!
    for (int i = mid; i < old_internal.count; ++i) {
        old_internal.keys[i] = 0;
    }
    for (int i = mid + 1; i <= old_internal.count; ++i) {
        old_internal.child_ids[i] = 0;
    }
    old_internal.count = static_cast<uint16_t>(mid);
    
    pager_.write_page(new_internal_id, new_internal);
    pager_.write_page(node_id, old_internal);
}

void BTree::insert_into_node(Page& node, Key key, Value val, PageID child_id) {
    // Shift elements right to maintain sorted order
    int i = node.count;
    
    if (node.type == NodeType::LEAF) {
        while (i > 0 && node.keys[i - 1] > key) {
            node.keys[i] = node.keys[i - 1];
            node.values[i] = node.values[i - 1];
            --i;
        }
        node.keys[i] = key;
        node.values[i] = val;
    } else {
        // For internal nodes: shift keys and the right child pointers
        while (i > 0 && node.keys[i - 1] > key) {
            node.keys[i] = node.keys[i - 1];
            node.child_ids[i + 1] = node.child_ids[i];
            --i;
        }
        node.keys[i] = key;
        node.child_ids[i + 1] = child_id;
    }
    node.count++;
}
