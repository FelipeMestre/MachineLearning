import { ComponentFixture, TestBed } from '@angular/core/testing';

import { WineDatasetComponent } from './wine-dataset.component';

describe('WineDatasetComponent', () => {
  let component: WineDatasetComponent;
  let fixture: ComponentFixture<WineDatasetComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ WineDatasetComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(WineDatasetComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
