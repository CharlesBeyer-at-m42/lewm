#include "lewm/gguf_store.hpp"

#include <cstdio>
#include <memory>
#include <stdexcept>

#include "ggml-alloc.h"

namespace lewm {

GgufStore::GgufStore(const std::string& path, ggml_backend_t backend) : path_(path) {
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &ctx_;
    gguf_ = gguf_init_from_file(path.c_str(), params);
    if (!gguf_) throw std::runtime_error("failed to open GGUF: " + path);

    buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend);
    if (!buffer_) throw std::runtime_error("failed to allocate tensors for: " + path);

    std::unique_ptr<FILE, int (*)(FILE*)> f(std::fopen(path.c_str(), "rb"), &std::fclose);
    if (!f) throw std::runtime_error("failed to reopen: " + path);

    const size_t data_offset = gguf_get_data_offset(gguf_);
    const int64_t n = gguf_get_n_tensors(gguf_);
    std::vector<char> buf;
    for (int64_t i = 0; i < n; ++i) {
        const char* name = gguf_get_tensor_name(gguf_, i);
        ggml_tensor* t = ggml_get_tensor(ctx_, name);
        if (!t) throw std::runtime_error(std::string("tensor missing from context: ") + name);
        const size_t nbytes = ggml_nbytes(t);
        buf.resize(nbytes);
        const long off = static_cast<long>(data_offset + gguf_get_tensor_offset(gguf_, i));
        if (std::fseek(f.get(), off, SEEK_SET) != 0 || std::fread(buf.data(), 1, nbytes, f.get()) != nbytes) {
            throw std::runtime_error(std::string("short read for tensor: ") + name);
        }
        ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
    }
}

GgufStore::~GgufStore() {
    if (buffer_) ggml_backend_buffer_free(buffer_);
    if (ctx_) ggml_free(ctx_);
    if (gguf_) gguf_free(gguf_);
}

ggml_tensor* GgufStore::find(const std::string& name) const {
    return ggml_get_tensor(ctx_, name.c_str());
}

ggml_tensor* GgufStore::tensor(const std::string& name) const {
    ggml_tensor* t = find(name);
    if (!t) throw std::runtime_error("tensor not found in " + path_ + ": " + name);
    return t;
}

std::vector<std::string> GgufStore::tensor_names() const {
    std::vector<std::string> out;
    const int64_t n = gguf_get_n_tensors(gguf_);
    out.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) out.emplace_back(gguf_get_tensor_name(gguf_, i));
    return out;
}

HostTensor GgufStore::read(const std::string& name) const {
    ggml_tensor* t = tensor(name);
    if (t->type != GGML_TYPE_F32) throw std::runtime_error("expected f32 tensor: " + name);
    HostTensor h;
    h.n_dims = ggml_n_dims(t);
    for (int i = 0; i < 4; ++i) h.ne[i] = t->ne[i];
    h.data.resize(static_cast<size_t>(ggml_nelements(t)));
    ggml_backend_tensor_get(t, h.data.data(), 0, ggml_nbytes(t));
    return h;
}

int64_t GgufStore::key_id(const std::string& key) const {
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0) throw std::runtime_error("metadata key not found in " + path_ + ": " + key);
    return id;
}

bool GgufStore::has_key(const std::string& key) const { return gguf_find_key(gguf_, key.c_str()) >= 0; }

uint32_t GgufStore::get_u32(const std::string& key) const { return gguf_get_val_u32(gguf_, key_id(key)); }

std::string GgufStore::get_str(const std::string& key) const { return gguf_get_val_str(gguf_, key_id(key)); }

}  // namespace lewm
