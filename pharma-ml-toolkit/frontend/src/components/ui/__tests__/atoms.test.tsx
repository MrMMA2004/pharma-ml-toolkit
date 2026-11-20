import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

/**
 * part13 Phase 3 — the shared UI atoms (the shadcn/ui primitives the
 * Mineral tokens theme). These are the building blocks every workspace
 * composes; their a11y contracts (real buttons, real headings, real
 * tablist wiring) are pinned here so a regression surfaces as a red
 * test, not as a silently degraded workspace.
 */

describe("Button", () => {
  it("renders a real button that fires onClick", async () => {
    const onClick = vi.fn();
    render(<Button onClick={onClick}>Run analysis</Button>);

    const button = screen.getByRole("button", { name: "Run analysis" });
    await userEvent.click(button);
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("applies the variant + size classes (the Mineral token contract)", () => {
    const { rerender } = render(<Button>default</Button>);
    expect(screen.getByRole("button")).toHaveClass(
      "bg-[var(--color-primary)]",
      "h-9",
    );

    rerender(<Button variant="destructive">destructive</Button>);
    expect(screen.getByRole("button")).toHaveClass(
      "bg-[var(--color-destructive)]",
    );

    rerender(<Button variant="outline" size="sm">outline</Button>);
    expect(screen.getByRole("button")).toHaveClass("border", "h-8", "text-xs");
  });

  it("stays honest when disabled: the attribute + no click dispatch", async () => {
    const onClick = vi.fn();
    render(
      <Button disabled onClick={onClick}>
        Upload
      </Button>,
    );
    const button = screen.getByRole("button", { name: "Upload" });
    expect(button).toBeDisabled();
    await userEvent.click(button);
    expect(onClick).not.toHaveBeenCalled();
  });
});

describe("Card", () => {
  it("composes the header/title/description/content structure", () => {
    render(
      <Card>
        <CardHeader>
          <CardTitle>Descriptor report</CardTitle>
          <CardDescription>computed over 128 molecules</CardDescription>
        </CardHeader>
        <CardContent>the table lane</CardContent>
      </Card>,
    );

    expect(screen.getByRole("heading", { level: 3, name: "Descriptor report" })).toBeInTheDocument();
    expect(screen.getByText("computed over 128 molecules")).toBeInTheDocument();
    expect(screen.getByText("the table lane")).toBeInTheDocument();
  });
});

describe("Tabs", () => {
  it("exposes the tablist semantics and switches content on click", async () => {
    render(
      <Tabs defaultValue="upload">
        <TabsList>
          <TabsTrigger value="upload">Upload</TabsTrigger>
          <TabsTrigger value="inspect">Inspect</TabsTrigger>
        </TabsList>
        <TabsContent value="upload">the import lane</TabsContent>
        <TabsContent value="inspect">the preview lane</TabsContent>
      </Tabs>,
    );

    const list = screen.getByRole("tablist");
    expect(list).toBeInTheDocument();
    const uploadTab = screen.getByRole("tab", { name: "Upload" });
    expect(uploadTab).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: "Inspect" })).toHaveAttribute(
      "aria-selected",
      "false",
    );
    expect(screen.getByText("the import lane")).toBeInTheDocument();
    expect(screen.queryByText("the preview lane")).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: "Inspect" }));
    expect(screen.getByText("the preview lane")).toBeInTheDocument();
    expect(screen.queryByText("the import lane")).not.toBeInTheDocument();
  });

  it("operates by keyboard — the arrow-key roving tabindex", async () => {
    render(
      <Tabs defaultValue="upload">
        <TabsList>
          <TabsTrigger value="upload">Upload</TabsTrigger>
          <TabsTrigger value="inspect">Inspect</TabsTrigger>
        </TabsList>
        <TabsContent value="upload">the import lane</TabsContent>
        <TabsContent value="inspect">the preview lane</TabsContent>
      </Tabs>,
    );

    screen.getByRole("tab", { name: "Upload" }).focus();
    await userEvent.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "Inspect" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(screen.getByText("the preview lane")).toBeInTheDocument();
  });
});
